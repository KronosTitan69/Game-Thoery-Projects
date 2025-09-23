"""
Tamil Nadu Election Forecasting - Visualization Module
=====================================================

Comprehensive visualization tools for election forecasting results,
model validation, and trend analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import networkx as nx
from election_forecasting_model import *

class ElectionVisualizationSuite:
    """Comprehensive visualization suite for election forecasting"""
    
    def __init__(self):
        self.color_palette = {
            'DMK': '#FF6B35',      # Orange-red (DMK colors)
            'AIADMK': '#4CAF50',   # Green (AIADMK colors) 
            'BJP': '#FF9800',      # Saffron (BJP colors)
            'Others': '#9E9E9E',
            'Undecided': '#E0E0E0'
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_demographic_profile(self, demographic_data: DemographicProfile, 
                                save_path: Optional[str] = None):
        """Visualize district demographic profile"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('District Demographic Profile', fontsize=16, fontweight='bold')
        
        # Age distribution
        ages = list(demographic_data.age_distribution.keys())
        age_props = list(demographic_data.age_distribution.values())
        axes[0, 0].pie(age_props, labels=ages, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Age Distribution')
        
        # Caste composition
        castes = list(demographic_data.caste_composition.keys())
        caste_props = list(demographic_data.caste_composition.values())
        bars = axes[0, 1].bar(castes, caste_props, color=sns.color_palette("Set2", len(castes)))
        axes[0, 1].set_title('Caste Composition')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Religious affiliation
        religions = list(demographic_data.religious_affiliation.keys())
        religion_props = list(demographic_data.religious_affiliation.values())
        axes[0, 2].pie(religion_props, labels=religions, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Religious Affiliation')
        
        # Occupation breakdown
        occupations = list(demographic_data.occupation_breakdown.keys())
        occ_props = list(demographic_data.occupation_breakdown.values())
        axes[1, 0].bar(occupations, occ_props, color=sns.color_palette("Set3", len(occupations)))
        axes[1, 0].set_title('Occupation Breakdown')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Urban-Rural split
        locations = list(demographic_data.rural_urban_split.keys())
        location_props = list(demographic_data.rural_urban_split.values())
        axes[1, 1].pie(location_props, labels=locations, autopct='%1.1f%%', startangle=90,
                      colors=['#8FBC8F', '#D2B48C'])
        axes[1, 1].set_title('Rural-Urban Split')
        
        # Gender ratio
        gender_data = ['Female', 'Male']
        gender_props = [demographic_data.gender_ratio, 1 - demographic_data.gender_ratio]
        axes[1, 2].pie(gender_props, labels=gender_data, autopct='%1.1f%%', startangle=90,
                      colors=['#FFB6C1', '#87CEEB'])
        axes[1, 2].set_title('Gender Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_voter_preference_evolution(self, behavioral_engine: AgentBehavioralEngine,
                                      save_path: Optional[str] = None):
        """Plot evolution of voter preferences over time"""
        
        if not behavioral_engine.preference_history:
            print("No preference history available. Run simulation first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Party preference evolution
        time_steps = range(len(behavioral_engine.preference_history))
        
        for party in ['DMK', 'AIADMK', 'BJP']:
            preferences = [step[party] for step in behavioral_engine.preference_history]
            ax1.plot(time_steps, preferences, marker='o', linewidth=2, 
                    label=party, color=self.color_palette[party])
        
        ax1.set_xlabel('Time Steps (Days)')
        ax1.set_ylabel('Average Party Preference')
        ax1.set_title('Evolution of Voter Preferences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Opinion state evolution
        opinion_states = ['Undecided', 'DMK Leaner', 'AIADMK Leaner', 'DMK Committed', 'AIADMK Committed']
        colors = ['#E0E0E0', '#FFB366', '#66B366', '#FF6B35', '#4CAF50']
        
        # Stack plot of opinion states
        state_data = []
        for state in VoterOpinionState:
            state_counts = [step[state] for step in behavioral_engine.opinion_state_history]
            state_data.append(state_counts)
        
        ax2.stackplot(time_steps, *state_data, labels=opinion_states, colors=colors, alpha=0.8)
        ax2.set_xlabel('Time Steps (Days)')
        ax2.set_ylabel('Number of Voters')
        ax2.set_title('Evolution of Opinion States')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_social_network(self, social_network: nx.Graph, population: List[SyntheticVoterAgent],
                           save_path: Optional[str] = None):
        """Visualize social network with voter preferences"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Network layout
        pos = nx.spring_layout(social_network, k=0.5, iterations=50)
        
        # Plot 1: Network colored by dominant party preference
        node_colors = []
        for agent in population:
            dominant_party = max(agent.party_preferences, key=agent.party_preferences.get)
            node_colors.append(self.color_palette[dominant_party])
        
        nx.draw_networkx(social_network, pos, ax=ax1, node_color=node_colors, 
                        node_size=30, alpha=0.8, with_labels=False, edge_color='gray', 
                        width=0.5)
        ax1.set_title('Social Network - Colored by Party Preference')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=self.color_palette[party], 
                                     markersize=10, label=party) 
                          for party in ['DMK', 'AIADMK', 'BJP']]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Network colored by demographic groups
        demo_colors = []
        caste_color_map = {'Forward_Caste': '#FF9999', 'OBC': '#66B2FF', 'SC': '#99FF99', 
                          'ST': '#FFCC99', 'Others': '#FF99FF'}
        
        for agent in population:
            demo_colors.append(caste_color_map.get(agent.caste, '#CCCCCC'))
        
        nx.draw_networkx(social_network, pos, ax=ax2, node_color=demo_colors,
                        node_size=30, alpha=0.8, with_labels=False, edge_color='gray',
                        width=0.5)
        ax2.set_title('Social Network - Colored by Caste Groups')
        
        # Create legend for demographics
        demo_legend = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color, markersize=10, label=caste)
                      for caste, color in caste_color_map.items()]
        ax2.legend(handles=demo_legend, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_forecasting_results(self, comprehensive_results: Dict[str, Any],
                               save_path: Optional[str] = None):
        """Visualize comprehensive forecasting results"""
        
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Deterministic forecast - Party vote shares
        ax1 = fig.add_subplot(gs[0, 0])
        det_forecast = comprehensive_results['deterministic_forecast']
        parties = list(det_forecast['party_vote_shares'].keys())
        vote_shares = list(det_forecast['party_vote_shares'].values())
        colors = [self.color_palette.get(party, '#999999') for party in parties]
        
        bars = ax1.bar(parties, vote_shares, color=colors, alpha=0.8)
        ax1.set_title('Deterministic Vote Share Forecast', fontweight='bold')
        ax1.set_ylabel('Vote Share')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, vote_shares):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Probabilistic forecast - Win probabilities
        ax2 = fig.add_subplot(gs[0, 1])
        prob_forecast = comprehensive_results['probabilistic_forecast']
        win_probs = prob_forecast['win_probabilities']
        
        parties_prob = list(win_probs.keys())
        probabilities = list(win_probs.values())
        colors_prob = [self.color_palette.get(party, '#999999') for party in parties_prob]
        
        bars2 = ax2.bar(parties_prob, probabilities, color=colors_prob, alpha=0.8)
        ax2.set_title('Win Probabilities', fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars2, probabilities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Vote share confidence intervals
        ax3 = fig.add_subplot(gs[0, 2])
        party_predictions = prob_forecast['party_predictions']
        
        parties_ci = list(party_predictions.keys())
        means = [party_predictions[party]['mean'] for party in parties_ci]
        ci_lower = [party_predictions[party]['confidence_interval_95'][0] for party in parties_ci]
        ci_upper = [party_predictions[party]['confidence_interval_95'][1] for party in parties_ci]
        colors_ci = [self.color_palette.get(party, '#999999') for party in parties_ci]
        
        x_pos = range(len(parties_ci))
        ax3.bar(x_pos, means, color=colors_ci, alpha=0.7)
        ax3.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower),
                                        np.array(ci_upper) - np.array(means)],
                    fmt='none', color='black', capsize=5, capthick=2)
        
        ax3.set_title('Vote Share Confidence Intervals (95%)', fontweight='bold')
        ax3.set_ylabel('Vote Share')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(parties_ci)
        
        # 4. Model validation metrics
        ax4 = fig.add_subplot(gs[1, 0])
        validation_metrics = comprehensive_results['validation_metrics']
        metrics = list(validation_metrics.keys())
        values = list(validation_metrics.values())
        
        # Filter out infinite values
        finite_metrics = [(m, v) for m, v in zip(metrics, values) if not np.isinf(v)]
        if finite_metrics:
            metrics, values = zip(*finite_metrics)
            ax4.bar(metrics, values, color='skyblue', alpha=0.8)
            ax4.set_title('Model Validation Metrics', fontweight='bold')
            ax4.set_ylabel('Error Value')
        else:
            ax4.text(0.5, 0.5, 'No finite validation metrics', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Validation Metrics', fontweight='bold')
        
        # 5. Candidate predictions
        ax5 = fig.add_subplot(gs[1, 1:])
        candidate_predictions = det_forecast['candidate_predictions']
        
        if candidate_predictions:
            candidates = list(candidate_predictions.keys())
            candidate_shares = [candidate_predictions[c]['predicted_vote_share'] 
                              for c in candidates]
            candidate_parties = [candidate_predictions[c]['party'] for c in candidates]
            candidate_colors = [self.color_palette.get(party, '#999999') 
                              for party in candidate_parties]
            
            bars5 = ax5.bar(candidates, candidate_shares, color=candidate_colors, alpha=0.8)
            ax5.set_title('Candidate-Level Predictions', fontweight='bold')
            ax5.set_ylabel('Predicted Vote Share')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add party labels
            for bar, party in zip(bars5, candidate_parties):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        party, ha='center', va='bottom', fontsize=10)
        
        # 6. Key insights text
        ax6 = fig.add_subplot(gs[2:, :])
        insights = comprehensive_results['key_insights']
        confidence = comprehensive_results['model_confidence']
        
        insight_text = f"Model Confidence: {confidence}\n\n"
        insight_text += "Key Insights:\n"
        for i, insight in enumerate(insights, 1):
            insight_text += f"{i}. {insight}\n"
        
        # Add summary statistics
        insight_text += f"\nForecast Summary:\n"
        insight_text += f"• Predicted Winner: {det_forecast['winning_party']}\n"
        insight_text += f"• Winning Margin: {det_forecast['winning_margin']:.3f}\n"
        insight_text += f"• Expected Turnout: {det_forecast['predicted_turnout']:.3f}\n"
        insight_text += f"• Most Likely Winner (Probabilistic): {prob_forecast['most_likely_winner']}\n"
        
        ax6.text(0.05, 0.95, insight_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightgray", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Election Forecast Summary & Insights', fontweight='bold', pad=20)
        
        plt.suptitle('Tamil Nadu Election Forecasting Results', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_markov_transitions(self, markov_model: MarkovProcessModel,
                              save_path: Optional[str] = None):
        """Visualize Markov transition matrices"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Voter opinion transitions
        voter_states = ['Undecided', 'DMK Leaner', 'AIADMK Leaner', 'DMK Committed', 'AIADMK Committed']
        
        im1 = ax1.imshow(markov_model.voter_transition_matrix, cmap='Blues', aspect='equal')
        ax1.set_xticks(range(len(voter_states)))
        ax1.set_yticks(range(len(voter_states)))
        ax1.set_xticklabels(voter_states, rotation=45, ha='right')
        ax1.set_yticklabels(voter_states)
        ax1.set_xlabel('To State')
        ax1.set_ylabel('From State')
        ax1.set_title('Voter Opinion Transition Matrix')
        
        # Add transition probabilities as text
        for i in range(len(voter_states)):
            for j in range(len(voter_states)):
                text = ax1.text(j, i, f'{markov_model.voter_transition_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white" if 
                               markov_model.voter_transition_matrix[i, j] > 0.5 else "black")
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Candidate career transitions
        career_states = ['New', 'Incumbent', 'Re-elected', 'Retired']
        
        im2 = ax2.imshow(markov_model.candidate_transition_matrix, cmap='Greens', aspect='equal')
        ax2.set_xticks(range(len(career_states)))
        ax2.set_yticks(range(len(career_states)))
        ax2.set_xticklabels(career_states)
        ax2.set_yticklabels(career_states)
        ax2.set_xlabel('To State')
        ax2.set_ylabel('From State')
        ax2.set_title('Candidate Career Transition Matrix')
        
        # Add transition probabilities as text
        for i in range(len(career_states)):
            for j in range(len(career_states)):
                text = ax2.text(j, i, f'{markov_model.candidate_transition_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white" if 
                               markov_model.candidate_transition_matrix[i, j] > 0.5 else "black")
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, forecasting_engine: ElectionForecastingEngine,
                        comprehensive_results: Dict[str, Any],
                        save_path: Optional[str] = None):
        """Create comprehensive dashboard with all visualizations"""
        
        print("🎨 Creating comprehensive election forecasting dashboard...")
        
        # Plot demographic profile
        print("📊 Generating demographic visualizations...")
        self.plot_demographic_profile(forecasting_engine.data_processor.demographic_data)
        
        # Create and run a behavioral simulation for visualization
        print("🧠 Running behavioral simulation for visualization...")
        pop_generator = SyntheticPopulationGenerator(forecasting_engine.data_processor.demographic_data)
        population = pop_generator.generate_population(1000)
        params = ModelParams(population_size=1000, time_horizon=30)
        behavioral_engine = AgentBehavioralEngine(population, pop_generator.social_network, params)
        
        # Run simulation
        for t in range(30):
            campaign_shocks = None
            if t == 10:
                campaign_shocks = {"DMK": 0.02, "AIADMK": -0.01}
            elif t == 20:
                campaign_shocks = {"AIADMK": 0.015, "BJP": 0.01}
            
            behavioral_engine.update_population_preferences(
                forecasting_engine.data_processor.socioeconomic_data,
                forecasting_engine.data_processor.candidates_data,
                campaign_shocks
            )
        
        # Plot preference evolution
        print("📈 Generating preference evolution plots...")
        self.plot_voter_preference_evolution(behavioral_engine)
        
        # Plot social network
        print("🕸️ Generating social network visualization...")
        self.plot_social_network(pop_generator.social_network, population)
        
        # Plot Markov transitions
        print("🔄 Generating Markov transition visualizations...")
        markov_model = MarkovProcessModel()
        self.plot_markov_transitions(markov_model)
        
        # Plot forecasting results
        print("🎯 Generating forecasting results dashboard...")
        self.plot_forecasting_results(comprehensive_results)
        
        print("✅ Dashboard generation complete!")


if __name__ == "__main__":
    print("🎨 Tamil Nadu Election Forecasting - Visualization Demo")
    print("=" * 60)
    
    # Initialize complete forecasting system
    data_processor = DataIngestionProcessor("Coimbatore")
    data_processor.load_demographic_data()
    data_processor.load_socioeconomic_indicators()
    data_processor.load_candidate_data()
    data_processor.load_historical_results()
    
    forecasting_engine = ElectionForecastingEngine(data_processor)
    
    # Run a quick forecast
    print("🎯 Running election forecast...")
    deterministic_forecast = forecasting_engine.generate_deterministic_forecast(60)
    probabilistic_forecast = forecasting_engine.generate_probabilistic_forecast(60, 20)
    
    comprehensive_results = {
        'validation_metrics': {'mape': 0.08, 'rmse': 0.06, 'mae': 0.05},
        'deterministic_forecast': deterministic_forecast,
        'probabilistic_forecast': probabilistic_forecast,
        'model_confidence': 'Medium',
        'key_insights': [
            'Competitive race between DMK and AIADMK',
            'BJP showing marginal presence',
            'High voter turnout expected'
        ]
    }
    
    # Create visualization suite
    viz_suite = ElectionVisualizationSuite()
    
    # Generate dashboard
    viz_suite.create_dashboard(forecasting_engine, comprehensive_results)