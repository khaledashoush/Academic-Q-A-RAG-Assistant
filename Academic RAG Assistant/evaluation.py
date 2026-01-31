"""
RAGAS evaluation module for Academic RAG Assistant
"""

import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_correctness,
    answer_similarity
)
from ragas.metrics import(
    harmfulness,
    maliciousness,
    coherence,
    correctness,
    conciseness
)
from ragas import evaluate
from datasets import Dataset

from config import config

class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics"""
    
    def __init__(self):
        self.metrics = self._initialize_metrics()
        self.evaluation_results = {}
    
    def _initialize_metrics(self):
        """Initialize RAGAS metrics based on config"""
        metrics = []
        
        metric_mapping = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'context_entity_recall': context_entity_recall,
            'answer_correctness': answer_correctness,
            'answer_similarity': answer_similarity,
            'harmfulness': harmfulness,
            'maliciousness': maliciousness,
            'coherence': coherence,
            'correctness': correctness,
            'conciseness': conciseness
        }
        
        for metric_name in config.EVALUATION_METRICS:
            if metric_name in metric_mapping:
                metrics.append(metric_mapping[metric_name])
        
        return metrics
    
    async def evaluate_single_query(self, 
                                   query: str, 
                                   answer: str, 
                                   contexts: List[str],
                                   ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a single query-answer pair"""
        
        data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts],
        }
        
        if ground_truth:
            data['ground_truth'] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                raise_exceptions=False
            )
            
            scores = result.to_pandas().iloc[0].to_dict()
            return scores
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return {}
    
    def evaluate_batch(self, 
                      queries: List[str], 
                      answers: List[str], 
                      contexts_list: List[List[str]],
                      ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a batch of queries"""
        
        data = {
            'question': queries,
            'answer': answers,
            'contexts': contexts_list,
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                raise_exceptions=False
            )
            
            df_results = result.to_pandas()
            summary = self._calculate_summary(df_results)
            
            self.evaluation_results = {
                'detailed': df_results.to_dict('records'),
                'summary': summary,
                'statistics': self._calculate_statistics(df_results)
            }
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Batch evaluation error: {str(e)}")
            return {}
    
    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary statistics from evaluation results"""
        summary = {}
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                summary[column] = {
                    'mean': df[column].mean(),
                    'median': df[column].median(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max()
                }
        
        return summary
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional statistics"""
        stats = {
            'total_queries': len(df),
            'metrics_evaluated': len(df.columns) - 1,  # Exclude question column
            'avg_scores': {},
            'performance_band': {}
        }
        
        for column in df.columns:
            if column != 'question' and df[column].dtype in ['float64', 'int64']:
                avg_score = df[column].mean()
                stats['avg_scores'][column] = avg_score
                
                # Categorize performance
                if avg_score >= 0.8:
                    stats['performance_band'][column] = "Excellent"
                elif avg_score >= 0.6:
                    stats['performance_band'][column] = "Good"
                elif avg_score >= 0.4:
                    stats['performance_band'][column] = "Fair"
                else:
                    stats['performance_band'][column] = "Needs Improvement"
        
        return stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return {}
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': {
                'metrics': config.EVALUATION_METRICS,
                'thresholds': {
                    'excellent': 0.8,
                    'good': 0.6,
                    'fair': 0.4
                }
            },
            'results': self.evaluation_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if 'summary' not in self.evaluation_results:
            return recommendations
        
        summary = self.evaluation_results['summary']
        
        for metric, scores in summary.items():
            mean_score = scores['mean']
            
            if metric == 'faithfulness' and mean_score < 0.7:
                recommendations.append(
                    "Improve faithfulness by ensuring answers are directly derived from context"
                )
            elif metric == 'answer_relevancy' and mean_score < 0.7:
                recommendations.append(
                    "Improve answer relevancy by better matching responses to query intent"
                )
            elif metric == 'context_precision' and mean_score < 0.6:
                recommendations.append(
                    "Improve retrieval precision by adjusting chunk size or embedding model"
                )
            elif metric == 'context_recall' and mean_score < 0.6:
                recommendations.append(
                    "Improve context recall by increasing retrieval count or using hybrid search"
                )
        
        return recommendations
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create Plotly visualizations for evaluation results"""
        if not self.evaluation_results or 'detailed' not in self.evaluation_results:
            return {}
        
        df = pd.DataFrame(self.evaluation_results['detailed'])
        
        visualizations = {}
        
        # Radar chart for metrics
        metrics = [col for col in df.columns if col != 'question']
        avg_scores = [df[metric].mean() for metric in metrics]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=avg_scores,
            theta=metrics,
            fill='toself',
            name='Average Scores'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="RAGAS Evaluation Metrics - Radar Chart"
        )
        
        visualizations['radar_chart'] = fig_radar
        
        # Bar chart for individual queries
        fig_bar = go.Figure()
        
        for metric in metrics[:5]:  # Limit to first 5 metrics for clarity
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=df['question'].str[:30],  # Truncate long questions
                y=df[metric]
            ))
        
        fig_bar.update_layout(
            barmode='group',
            title="Per-Query Metric Scores",
            xaxis_title="Questions",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        visualizations['bar_chart'] = fig_bar
        
        # Heatmap for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig_heatmap.update_layout(
                title="Metric Correlation Heatmap",
                xaxis_title="Metrics",
                yaxis_title="Metrics"
            )
            
            visualizations['heatmap'] = fig_heatmap
        
        return visualizations