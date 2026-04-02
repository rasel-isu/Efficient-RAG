import json
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# Check imports and provide helpful error messages
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️  BERTScore not available. Install with: pip install bert-score")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class ComprehensiveRAGEvaluator:
    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu else "cpu"
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None

        if SENTENCE_TRANSFORMER_AVAILABLE:
            print("Loading sentence transformer model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        else:
            self.semantic_model = None
    
    def normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison"""
        text = text.lower().strip()
        text = ' '.join(text.split())
        text = re.sub(r'[.,;!?]+$', '', text)
        return text
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if answers match exactly (case-insensitive)"""
        return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)
    
    def partial_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer contains or is contained in ground truth"""
        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)
        return pred_norm in gt_norm or gt_norm in pred_norm
    
    def calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-level F1 score"""
        pred_tokens = self.normalize_answer(predicted).split()
        gt_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        common_tokens = set(pred_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def classify_question_type(self, question: str) -> str:
        """Classify question into types"""
        question_lower = question.lower()
        
        if any(question_lower.startswith(prefix) for prefix in 
               ['did ', 'was ', 'is ', 'were ', 'are ', 'does ', 'do ', 'can ', 
                'could ', 'would ', 'should ', 'has ', 'have ']):
            return 'yes_no'
        elif question_lower.startswith('when '):
            return 'when'
        elif question_lower.startswith('who '):
            return 'who'
        elif question_lower.startswith('where '):
            return 'where'
        elif question_lower.startswith('what '):
            return 'what'
        elif question_lower.startswith('how '):
            return 'how'
        elif question_lower.startswith('which '):
            return 'which'
        elif question_lower.startswith('why '):
            return 'why'
        else:
            return 'other'
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate BERTScore for semantic similarity"""
        if not BERTSCORE_AVAILABLE:
            return {'error': 'BERTScore not available'}
        
        print("Calculating BERTScore...")
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean()),
            'precision_std': float(P.std()),
            'recall_std': float(R.std()),
            'f1_std': float(F1.std()),
        }
    
    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {'error': 'ROUGE not available'}
        
        print("Calculating ROUGE scores...")
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': {
                'mean': float(np.mean(rouge1_scores)),
                'std': float(np.std(rouge1_scores)),
            },
            'rouge2': {
                'mean': float(np.mean(rouge2_scores)),
                'std': float(np.std(rouge2_scores)),
            },
            'rougeL': {
                'mean': float(np.mean(rougeL_scores)),
                'std': float(np.std(rougeL_scores)),
            },
        }
    
    def calculate_semantic_similarity(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate semantic similarity using sentence transformers"""
        if not SENTENCE_TRANSFORMER_AVAILABLE or self.semantic_model is None:
            return {'error': 'Sentence Transformers not available'}
        
        print("Calculating semantic similarity...")
        pred_embeddings = self.semantic_model.encode(predictions, convert_to_tensor=True)
        ref_embeddings = self.semantic_model.encode(references, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarities = util.cos_sim(pred_embeddings, ref_embeddings).diagonal()
        
        return {
            'mean': float(similarities.mean()),
            'std': float(similarities.std()),
            'min': float(similarities.min()),
            'max': float(similarities.max()),
        }
    
    def calculate_ragas_metrics(self, results: List[Dict]) -> Dict:
        """Calculate RAGAS metrics (requires contexts in data)"""
        if not RAGAS_AVAILABLE:
            return {'error': 'RAGAS not available'}
        
        # Check if we have contexts
        if not results or 'contexts' not in results[0] and 'retrieved_contexts' not in results[0]:
            return {
                'error': 'RAGAS requires retrieved contexts. Add "contexts" or "retrieved_contexts" field to your data.',
                'note': 'Example: {"question": "...", "answer": "...", "rag_answer": "...", "contexts": ["context1", "context2"]}'
            }
        
        print("Calculating RAGAS metrics...")
        
        # Prepare data for RAGAS
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for item in results:
            data["question"].append(item['question'])
            data["answer"].append(item['rag_answer'])
            # Try both field names
            contexts = item.get('contexts') or item.get('retrieved_contexts') or []
            # Ensure contexts is a list
            if isinstance(contexts, str):
                contexts = [contexts]
            data["contexts"].append(contexts)
            data["ground_truth"].append(item['answer'])
        
        try:
            dataset = Dataset.from_dict(data)
            
            result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
            )
            
            return {
                'context_precision': float(result.get('context_precision', 0)),
                'context_recall': float(result.get('context_recall', 0)),
                'faithfulness': float(result.get('faithfulness', 0)),
                'answer_relevancy': float(result.get('answer_relevancy', 0)),
            }
        except Exception as e:
            return {'error': f'RAGAS evaluation failed: {str(e)}'}
    
    def evaluate(self, results_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Comprehensive evaluation of RAG results
        
        Args:
            results_path: Path to JSON file with RAG results
            output_path: Optional path to save evaluation results
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RAG EVALUATION")
        print(f"{'='*80}\n")
        
        # Load results
        print(f"Loading results from: {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results\n")
        
        # Initialize metrics storage
        total = len(results)
        exact_matches = 0
        partial_matches = 0
        f1_scores = []
        
        predictions = []
        references = []
        
        # Token statistics
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []
        
        # By question type
        by_type = defaultdict(lambda: {'total': 0, 'exact_match': 0, 'f1_scores': []})
        
        # Track errors
        errors = []
        
        # Process each result
        print("Processing results...")
        for item in results:
            question = item['question']
            answer = item['answer']
            rag_answer = item['rag_answer']
            
            predictions.append(rag_answer)
            references.append(answer)
            
            # Question type
            q_type = self.classify_question_type(question)
            by_type[q_type]['total'] += 1
            
            # Exact match
            if self.exact_match(rag_answer, answer):
                exact_matches += 1
                by_type[q_type]['exact_match'] += 1
            else:
                errors.append({
                    'question': question,
                    'expected': answer,
                    'got': rag_answer,
                    'type': q_type
                })
            
            # Partial match
            if self.partial_match(rag_answer, answer):
                partial_matches += 1
            
            # F1 score
            f1 = self.calculate_f1_score(rag_answer, answer)
            f1_scores.append(f1)
            by_type[q_type]['f1_scores'].append(f1)
            
            # Token stats
            if 'prompt_token' in item:
                prompt_tokens.append(item['prompt_token'])
            if 'completion_token' in item:
                completion_tokens.append(item['completion_token'])
            if 'total_token' in item:
                total_tokens.append(item['total_token'])
        
        # Calculate aggregate metrics
        print("\n" + "="*80)
        print("CALCULATING METRICS")
        print("="*80 + "\n")
        
        metrics = {
            "experiment_name":results_path.split('/')[-1].replace('.json', ''),
            'summary': {
                'total_questions': total,
                'evaluation_date': str(np.datetime64('now')),
            },
            'basic_metrics': {
                'exact_match_accuracy': exact_matches / total,
                'partial_match_accuracy': partial_matches / total,
                'mean_f1_score': float(np.mean(f1_scores)),
                'median_f1_score': float(np.median(f1_scores)),
                'std_f1_score': float(np.std(f1_scores)),
            },
            'by_question_type': {},
            'error_analysis': {
                'total_errors': len(errors),
                'error_rate': len(errors) / total,
                'sample_errors': errors[:20]  # First 20 errors
            }
        }
        
        # Add token statistics if available
        if prompt_tokens:
            metrics['token_statistics'] = {
                'avg_prompt_tokens': float(np.mean(prompt_tokens)),
                'avg_completion_tokens': float(np.mean(completion_tokens)),
                'avg_total_tokens': float(np.mean(total_tokens)),
                'total_prompt_tokens': int(sum(prompt_tokens)),
                'total_completion_tokens': int(sum(completion_tokens)),
                'total_tokens_used': int(sum(total_tokens)),
                'median_prompt_tokens': float(np.median(prompt_tokens)),
                'median_completion_tokens': float(np.median(completion_tokens)),
            }
        
        # Add per-type metrics
        for q_type, stats in by_type.items():
            if stats['total'] > 0:
                metrics['by_question_type'][q_type] = {
                    'count': stats['total'],
                    'exact_match_accuracy': stats['exact_match'] / stats['total'],
                    'mean_f1_score': float(np.mean(stats['f1_scores'])) if stats['f1_scores'] else 0,
                }
        
        # Calculate BERTScore
        bert_scores = self.calculate_bert_score(predictions, references)
        metrics['bert_score'] = bert_scores
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        metrics['rouge_scores'] = rouge_scores
        
        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(predictions, references)
        metrics['semantic_similarity'] = semantic_sim
        
        # Calculate RAGAS metrics
        ragas_metrics = self.calculate_ragas_metrics(results)
        metrics['ragas_metrics'] = ragas_metrics
        
        # Print summary
        self.print_evaluation_summary(metrics)
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n Evaluation results saved to: {output_path}")
        
        return metrics
    
    def print_evaluation_summary(self, metrics: Dict):
        """Print formatted evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Basic metrics
        print("\n BASIC METRICS:")
        basic = metrics['basic_metrics']
        print(f"  Total Questions: {metrics['summary']['total_questions']}")
        print(f"  Exact Match Accuracy: {basic['exact_match_accuracy']:.2%}")
        print(f"  Partial Match Accuracy: {basic['partial_match_accuracy']:.2%}")
        print(f"  Mean F1 Score: {basic['mean_f1_score']:.4f}")
        print(f"  Median F1 Score: {basic['median_f1_score']:.4f}")
        
        # BERTScore
        if 'error' not in metrics['bert_score']:
            print("\n BERTSCORE (Semantic Similarity):")
            bert = metrics['bert_score']
            print(f"  Precision: {bert['precision']:.4f} (±{bert['precision_std']:.4f})")
            print(f"  Recall: {bert['recall']:.4f} (±{bert['recall_std']:.4f})")
            print(f"  F1: {bert['f1']:.4f} (±{bert['f1_std']:.4f})")
        
        # ROUGE scores
        if 'error' not in metrics['rouge_scores']:
            print("\n ROUGE SCORES:")
            rouge = metrics['rouge_scores']
            print(f"  ROUGE-1: {rouge['rouge1']['mean']:.4f} (±{rouge['rouge1']['std']:.4f})")
            print(f"  ROUGE-2: {rouge['rouge2']['mean']:.4f} (±{rouge['rouge2']['std']:.4f})")
            print(f"  ROUGE-L: {rouge['rougeL']['mean']:.4f} (±{rouge['rougeL']['std']:.4f})")
        
        # Semantic similarity
        if 'error' not in metrics['semantic_similarity']:
            print("\n SEMANTIC SIMILARITY (Sentence Transformers):")
            sem = metrics['semantic_similarity']
            print(f"  Mean Cosine Similarity: {sem['mean']:.4f} (±{sem['std']:.4f})")
            print(f"  Range: [{sem['min']:.4f}, {sem['max']:.4f}]")
        
        # RAGAS metrics
        if 'error' not in metrics['ragas_metrics']:
            print("\n RAGAS METRICS:")
            ragas = metrics['ragas_metrics']
            print(f"  Context Precision: {ragas['context_precision']:.4f}")
            print(f"  Context Recall: {ragas['context_recall']:.4f}")
            print(f"  Faithfulness: {ragas['faithfulness']:.4f}")
            print(f"  Answer Relevancy: {ragas['answer_relevancy']:.4f}")
        else:
            print("\nRAGAS METRICS:")
            print(f"  {metrics['ragas_metrics']['error']}")
        
        # Token statistics
        if 'token_statistics' in metrics:
            print("\nTOKEN USAGE:")
            tokens = metrics['token_statistics']
            print(f"  Average Tokens/Question: {tokens['avg_total_tokens']:.1f}")
            print(f"  Total Tokens Used: {tokens['total_tokens_used']:,}")
        
        # Performance by question type
        print("PERFORMANCE BY QUESTION TYPE:")
        sorted_types = sorted(
            metrics['by_question_type'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        for q_type, stats in sorted_types[:5]:  # Top 5
            print(f"\n  {q_type.upper()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Accuracy: {stats['exact_match_accuracy']:.2%}")
            print(f"    F1: {stats['mean_f1_score']:.4f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive RAG Evaluation with Multiple Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
Examples:
  # Basic evaluation
  python comprehensive_rag_eval.py --input baseline_rag.json
  
  # Save results to specific file
  python comprehensive_rag_eval.py --input baseline_rag.json --output results.json
  
  # Use GPU for faster computation
  python comprehensive_rag_eval.py --input baseline_rag.json --use-gpu

Required JSON format:
  [
    {
      "question": "What is X?",
      "answer": "Y",
      "rag_answer": "Y is...",
      "prompt_token": 100,      // optional
      "completion_token": 10,   // optional
      "total_token": 110,       // optional
      "contexts": ["..."]       // optional, required for RAGAS
    },
    ...
  ]
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input JSON file with RAG results'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='evaluation_results.json',
        help='Path to output JSON file for evaluation metrics (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for computations if available'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run evaluation
    evaluator = ComprehensiveRAGEvaluator(use_gpu=args.use_gpu)
    evaluator.evaluate(args.input, args.output)


if __name__ == '__main__':
    main()



# python comprehensive_rag_eval.py --input OUTPUT/rag-mini-wikipedia/gpt-3.5-turbo/t5_large_sumry.json --output OUTPUT/rag-mini-wikipedia/gpt-3.5-turbo/t5_large_sumry_performence.json > OUTPUT/rag-mini-wikipedia/gpt-3.5-turbo/t5_large_sumry_performence.log 
