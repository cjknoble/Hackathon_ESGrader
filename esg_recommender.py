from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum

class ProjectType(Enum):
    SOLAR = "solar"
    WIND = "wind"
    GEOTHERMAL = "geothermal"
    BIOMASS = "biomass"
    GAS = "gas"

class Timeline(Enum):
    SHORT = "short-term"
    MEDIUM = "medium-term"
    LONG = "long-term"

@dataclass
class Partner:
    name: str
    type: str
    description: str

@dataclass
class Impact:
    esg: float
    score: float

@dataclass
class Recommendation:
    title: str
    description: str
    impact: Impact
    timeline: Timeline
    category: str
    partners: Optional[List[Partner]] = None
    priority: Optional[float] = None


class ESGRecommendationEngine:
    def __init__(self):

        # ESG Dimension Weightings by Project Type
        self.esg_weightings = {
            "environmental": {
                "ghg_emissions": {
                    ProjectType.BIOMASS: 0.40,
                    ProjectType.GAS: 0.30,
                    ProjectType.SOLAR: 0.50,
                    ProjectType.WIND: 0.50,
                    ProjectType.GEOTHERMAL: 0.55
                },
                "biodiversity_land_water": {
                    ProjectType.BIOMASS: 0.30,
                    ProjectType.GAS: 0.35,
                    ProjectType.SOLAR: 0.20,
                    ProjectType.WIND: 0.15,
                    ProjectType.GEOTHERMAL: 0.20
                },
                "circular_economy": {
                    "default": 0.10  # Equal across all
                }
            },
            "social": {
                "energy_access": {
                    ProjectType.BIOMASS: 0.10,
                    ProjectType.GAS: 0.15,
                    ProjectType.SOLAR: 0.25,
                    ProjectType.WIND: 0.30,
                    ProjectType.GEOTHERMAL: 0.20
                },
                "job_creation": {
                    ProjectType.BIOMASS: 0.15,
                    ProjectType.GAS: 0.10,
                    ProjectType.SOLAR: 0.10,
                    ProjectType.WIND: 0.10,
                    ProjectType.GEOTHERMAL: 0.10
                },
                "equitable_access": {
                    ProjectType.BIOMASS: 0.05,
                    ProjectType.GAS: 0.10,
                    ProjectType.SOLAR: 0.05,
                    ProjectType.WIND: 0.10,
                    ProjectType.GEOTHERMAL: 0.10
                }
            },
            "governance": {
                "transparency": {
                    ProjectType.BIOMASS: 0.15,
                    ProjectType.GAS: 0.10,
                    ProjectType.SOLAR: 0.10,
                    ProjectType.WIND: 0.05,
                    ProjectType.GEOTHERMAL: 0.05
                },
                "compliance": {
                    ProjectType.BIOMASS: 0.10,
                    ProjectType.GAS: 0.05,
                    ProjectType.SOLAR: 0.05,
                    ProjectType.WIND: 0.05,
                    ProjectType.GEOTHERMAL: 0.05
                },
                "stakeholder_engagement": {
                    "default": 0.05  # Equal across all
                }
            }
        }

        # Impact Score Weightings by Project Type
        self.impact_weightings = {
            "climate_change_mitigation": {
                "ghg_reduction": {
                    ProjectType.BIOMASS: 0.35,
                    ProjectType.GAS: 0.20,
                    ProjectType.SOLAR: 0.40,
                    ProjectType.WIND: 0.45,
                    ProjectType.GEOTHERMAL: 0.50
                },
                "operational_emissions": {
                    ProjectType.BIOMASS: 0.15,
                    ProjectType.GAS: 0.10,
                    ProjectType.SOLAR: 0.20,
                    ProjectType.WIND: 0.25,
                    ProjectType.GEOTHERMAL: 0.30
                }
            },
            "local_environmental": {
                "biodiversity": {
                    ProjectType.BIOMASS: 0.25,
                    ProjectType.GAS: 0.30,
                    ProjectType.SOLAR: 0.20,
                    ProjectType.WIND: 0.15,
                    ProjectType.GEOTHERMAL: 0.20
                },
                "pollution_risks": {
                    ProjectType.BIOMASS: 0.20,
                    ProjectType.GAS: 0.20,
                    ProjectType.SOLAR: 0.10,
                    ProjectType.WIND: 0.10,
                    ProjectType.GEOTHERMAL: 0.10
                }
            },
            "community_benefits": {
                "jobs_economy": {
                    ProjectType.BIOMASS: 0.20,
                    ProjectType.GAS: 0.25,
                    ProjectType.SOLAR: 0.25,
                    ProjectType.WIND: 0.30,
                    ProjectType.GEOTHERMAL: 0.20
                },
                "energy_access_cost": {
                    ProjectType.BIOMASS: 0.10,
                    ProjectType.GAS: 0.15,
                    ProjectType.SOLAR: 0.15,
                    ProjectType.WIND: 0.20,
                    ProjectType.GEOTHERMAL: 0.20
                }
            },
            "health_safety": {
                "public_health_risks": {
                    ProjectType.BIOMASS: 0.10,
                    ProjectType.GAS: 0.20,
                    ProjectType.SOLAR: 0.10,
                    ProjectType.WIND: 0.05,
                    ProjectType.GEOTHERMAL: 0.05
                }
            },
            "innovation": {
                "technology_scalability": {
                    "default": 0.05  # Equal across all
                }
            }
        }

        # Question mappings to ESG categories
        self.question_mappings = {
            "environmental": {
                "ghg_emissions": [1, 6, 9],
                "biodiversity_land_water": [7, 22, 23],
                "circular_economy": [24, 25]
            },
            "social": {
                "energy_access": [2, 8],
                "job_creation": [3, 26, 27],
                "equitable_access": [13, 14]
            },
            "governance": {
                "transparency": [4, 11],
                "compliance": [5, 12, 15],
                "stakeholder_engagement": [28, 29, 30, 31]
            }
        }

        # Question mappings to Impact categories
        self.question_impact_mappings = {
            "climate_change_mitigation": {
                "ghg_reduction": [9, 19, 23],
                "operational_emissions": [1, 6, 24]
            },
            "local_environmental": {
                "biodiversity": [7, 22],
                "pollution_risks": [24]
            },
            "community_benefits": {
                "jobs_economy": [2, 3, 26],
                "energy_access_cost": [20, 32, 33]
            },
            "health_safety": {
                "public_health_risks": [10, 25]
            },
            "innovation": {
                "technology_scalability": [21, 28]
            }
        }

    def calculate_base_scores(self, survey_responses: pd.DataFrame, project_type: ProjectType) -> Dict[str, float]:
        """Calculate base ESG scores using the detailed weighting matrix."""
        scores = {}

        for category, subcategories in self.question_mappings.items():
            category_score = 0
            for subcategory, questions in subcategories.items():
                # Get responses for this subcategory
                subcategory_responses = survey_responses[survey_responses['question_id'].isin(questions)]

                # Calculate subcategory score
                if subcategory_responses.empty:
                    continue

                subcategory_score = subcategory_responses['response'].map({'Yes': 1, 'No': 0}).mean() * 100

                # Get weight for this subcategory and project type
                weight = self.esg_weightings[category][subcategory].get(project_type,
                                                                        self.esg_weightings[category][subcategory].get(
                                                                            'default', 0))

                category_score += subcategory_score * weight

            scores[category] = category_score

        return scores

    def calculate_impact_scores(self, survey_responses: pd.DataFrame, project_type: ProjectType) -> Dict[str, float]:
        """Calculate impact scores using the impact weighting matrix."""
        impact_scores = {}

        for category, subcategories in self.question_impact_mappings.items():
            category_score = 0
            for subcategory, questions in subcategories.items():
                # Get responses for this subcategory
                subcategory_responses = survey_responses[survey_responses['question_id'].isin(questions)]

                # Calculate subcategory score
                if subcategory_responses.empty:
                    continue

                subcategory_score = subcategory_responses['response'].map({'Yes': 1, 'No': 0}).mean() * 100

                # Get weight for this subcategory and project type
                weight = self.impact_weightings[category][subcategory].get(project_type,
                                                                           self.impact_weightings[category][
                                                                               subcategory].get('default', 0))

                category_score += subcategory_score * weight

            impact_scores[category] = category_score

        return impact_scores

    def apply_project_modifiers(self, scores: Dict[str, float], project_type: ProjectType) -> Dict[str, float]:
        """Apply project-specific modifiers based on the weighting matrices."""
        modified_scores = scores.copy()

        # Apply modifiers based on project type characteristics
        for category in modified_scores:
            if category == "environmental":
                if project_type in [ProjectType.SOLAR, ProjectType.WIND]:
                    modified_scores[category] *= 1.1  # Higher weight for renewable
                elif project_type == ProjectType.BIOMASS:
                    modified_scores[category] *= 0.9  # Lower due to emissions

            elif category == "social":
                if project_type in [ProjectType.WIND, ProjectType.SOLAR]:
                    modified_scores[category] *= 1.2  # Higher for job creation potential

        return modified_scores

    def incorporate_location_context(self, scores: Dict[str, float], location_data: pd.DataFrame) -> Dict[str, float]:
        """Adjust scores based on demographic and environmental justice data."""
        contextual_scores = scores.copy()

        if 'demographic_index' in location_data.columns:
            demographic_modifier = location_data['demographic_index'].iloc[0] / 100
            contextual_scores['social'] *= (1 + (demographic_modifier - 0.5))

        if 'environmental_justice_score' in location_data.columns:
            ej_modifier = location_data['environmental_justice_score'].iloc[0] / 100
            contextual_scores['environmental'] *= (1 + (ej_modifier - 0.5))

        return contextual_scores

    def generate_recommendations(self, scores: Dict[str, float], impact_scores: Dict[str, float],
                                 project_type: ProjectType) -> List[Recommendation]:
        """Generate recommendations based on both ESG and impact scores."""
        recommendations = []

        # Environmental recommendations based on scores and impact assessment
        if scores['environmental'] < 60 or impact_scores.get('climate_change_mitigation', 0) < 50:
            recommendations.append(
                Recommendation(
                    title="Implement Advanced Environmental Monitoring",
                    description="Deploy comprehensive environmental monitoring systems with focus on GHG emissions and resource usage optimization.",
                    impact=Impact(esg=15, score=20),
                    timeline=Timeline.MEDIUM,
                    category="environmental",
                    partners=[
                        Partner(
                            name="GreenTech Certifications",
                            type="certifier",
                            description="Leading certification body specializing in environmental compliance"
                        ),
                        Partner(
                            name="EcoInnovate Solutions",
                            type="technology",
                            description="Cutting-edge environmental monitoring provider"
                        )
                    ]
                )
            )
            # Social recommendations
            if scores['social'] < 50 or impact_scores.get('community_benefits', 0) < 50:
                recommendations.append(
                    Recommendation(
                        title="Enhance Community Engagement Initiatives",
                        description="Develop programs to improve community engagement, focusing on inclusivity, diversity, and equitable access to resources.",
                        impact=Impact(esg=20, score=25),
                        timeline=Timeline.LONG,
                        category="social",
                        partners=[
                            Partner(
                                name="Community Builders Network",
                                type="nonprofit",
                                description="Organization dedicated to fostering community development and engagement."
                            ),
                            Partner(
                                name="Social Equity Innovations",
                                type="consulting",
                                description="Experts in designing and implementing equitable social programs."
                            )
                        ]
                    )
                )
                # Governance recommendations
                if scores['governance'] < 50 or impact_scores.get('health_safety', 0) < 50:
                    recommendations.append(
                        Recommendation(
                            title="Strengthen Governance and Safety Frameworks",
                            description="Implement robust governance practices and enhance health and safety standards to ensure organizational resilience and compliance.",
                            impact=Impact(esg=18, score=22),
                            timeline=Timeline.MEDIUM,
                            category="governance",
                            partners=[
                                Partner(
                                    name="SafetyFirst Compliance",
                                    type="certifier",
                                    description="Specialists in health and safety certifications and risk management."
                                ),
                                Partner(
                                    name="GovernWell Consulting",
                                    type="consulting",
                                    description="Advisors in corporate governance and ethical practices."
                                )
                            ]
                        )
                    )

        # Add more recommendation logic based on scores and project type...

        # Calculate priorities considering both ESG and impact scores
        for rec in recommendations:
            rec.priority = self._calculate_priority(scores[rec.category],
                                                    impact_scores.get(rec.category.lower(), 0),
                                                    rec.impact)

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority or 0, reverse=True)

        return recommendations

    def _calculate_priority(self, category_score: float, impact_score: float,
                            recommendation_impact: Impact) -> float:
        """Calculate priority score considering both ESG and impact metrics."""
        urgency = 100 - ((category_score + impact_score) / 2)
        effectiveness_ratio = recommendation_impact.score / recommendation_impact.esg
        return (urgency * 0.7) + (effectiveness_ratio * 0.3)

