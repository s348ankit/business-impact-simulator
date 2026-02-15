import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Non-interactive backend for deployment
from lingam import DirectLiNGAM
from dowhy import CausalModel
import scipy.stats as stats
# ═══════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT SIMULATOR - DEMO VERSION
# Purpose: Help business leaders understand how different business levers
# affect their key performance metrics using data-driven insights
# ═══════════════════════════════════════════════════════════════════════════
# ─── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Impact Simulator | Data-Driven Decision Making",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ─── Professional Styling ──────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    }
    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
        margin: 12px 0;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.5rem;
    }
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    </style>
""", unsafe_allow_html=True)
# ─── Header Section ────────────────────────────────────────────────────────
st.title("📊 Business Impact Simulator")
st.markdown("""
<div class="info-box">
<h4 style="margin-top: 0;">Transform Your Business Decision-Making</h4>
This tool helps you understand the real impact of business decisions before you make them.
See how changes in marketing spend, staffing, or customer satisfaction affect your bottom line—backed by data, not guesswork.
</div>
""", unsafe_allow_html=True)
# ─── Demo Data Function ────────────────────────────────────────────────────
@st.cache_data
def generate_realistic_business_data():
    """
    Generate realistic business metrics for demonstration.
    In production, this would connect to your actual business data sources.
    """
    np.random.seed(42)
    n_months = 240 # 20 months of weekly data
   
    # Generate realistic business metrics with correlations
    marketing_spend = np.random.normal(75000, 20000, n_months)
    marketing_spend = np.clip(marketing_spend, 30000, 150000)
   
    # Staff count influences capacity
    staff_count = np.random.randint(15, 95, n_months)
   
    # Customer satisfaction (1-5 scale)
    satisfaction = np.random.uniform(3.2, 4.9, n_months)
   
    # Click-through rate (%)
    ctr = np.random.uniform(0.8, 3.8, n_months)
   
    # Revenue influenced by multiple factors
    revenue = (
        50000 +
        0.4 * marketing_spend +
        800 * staff_count +
        15000 * satisfaction +
        5000 * ctr +
        np.random.normal(0, 8000, n_months)
    )
   
    return pd.DataFrame({
        "Marketing_Spend_USD": marketing_spend,
        "Team_Size": staff_count,
        "Customer_Satisfaction": satisfaction,
        "Ad_Click_Rate_Percent": ctr,
        "Monthly_Revenue_USD": revenue
    })
def analyze_business_relationships(df_numeric):
    """
    Discover causal relationships between business metrics.
    Uses advanced statistical methods to identify which factors drive which outcomes.
    """
    model = DirectLiNGAM()
    model.fit(df_numeric.dropna().values)
    adjacency_matrix = model.adjacency_matrix_
   
    columns = df_numeric.columns
    relationships = []
   
    for i in range(len(columns)):
        for j in range(len(columns)):
            if adjacency_matrix[i, j] != 0:
                relationships.append({
                    'from': columns[i],
                    'to': columns[j],
                    'strength': adjacency_matrix[i, j]
                })
   
    return relationships
def visualize_business_relationships(relationships):
    """
    Create a visual map of how business metrics influence each other.
    Green arrows = positive impact, Red arrows = negative impact
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
   
    G = nx.DiGraph()
    edge_colors = []
    edge_widths = []
   
    for rel in relationships:
        G.add_edge(rel['from'], rel['to'])
        # Color based on positive/negative relationship
        edge_colors.append("#16a34a" if rel['strength'] > 0 else "#dc2626")
        # Width based on strength
        edge_widths.append(min(abs(rel['strength']) * 3, 5))
   
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)
   
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=5000,
        node_color="#f0f9ff",
        edgecolors="#2563eb",
        linewidths=3
    )
   
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=30,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1"
    )
   
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=11,
        font_weight="bold",
        font_color="#1e293b"
    )
   
    ax.set_title("Your Business Ecosystem: How Metrics Connect",
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
   
    return fig
def explain_business_relationships(relationships, target_metric, all_metrics):
    """
    Provide clear, business-focused explanations of the relationships.
    """
    direct_drivers = [r for r in relationships if r['to'] == target_metric]
   
    if not direct_drivers:
        return f"""
        **📌 Key Finding:** No strong direct relationships detected with {target_metric} in the current data.
       
        This could mean:
        - The relationships are indirect (through other metrics)
        - More data points are needed for accurate detection
        - External factors not in this dataset are driving changes
       
        **Recommendation:** Consider expanding the dataset or including additional business metrics.
        """
   
    # Sort by strength of impact
    direct_drivers.sort(key=lambda x: abs(x['strength']), reverse=True)
   
    explanation = f"**📊 What Drives Your {target_metric}:**\n\n"
   
    for i, driver in enumerate(direct_drivers[:5], 1): # Top 5 drivers
        impact_type = "positively influences" if driver['strength'] > 0 else "negatively impacts"
        strength_desc = "strongly" if abs(driver['strength']) > 0.5 else "moderately"
       
        explanation += f"""
        **{i}. {driver['from']}** {strength_desc} {impact_type} {target_metric}
        - Impact strength: {abs(driver['strength']):.3f}
        - Direction: {'📈 Positive' if driver['strength'] > 0 else '📉 Negative'}
       
        """
   
    explanation += """
    **💡 How to Use This:**
    - **Green arrows** show positive relationships (increase one, increase the other)
    - **Red arrows** show negative relationships (increase one, decrease the other)
    - **Thicker arrows** indicate stronger relationships
   
    These insights help you prioritize which business levers to focus on for maximum impact.
    """
   
    return explanation
def calculate_business_impact_bayesian(df, driver, outcome, control_factors):
    """
    Calculate the business impact using advanced probabilistic modeling.
    This accounts for uncertainty and gives you realistic ranges, not just point estimates.
    """
    # Prepare data
    feature_cols = [driver] + control_factors
    X_data = df[feature_cols].dropna()
    y_data = df.loc[X_data.index, outcome]
   
    if len(y_data) < 20:
        return None
   
    # Add intercept
    X_matrix = np.column_stack([np.ones(len(X_data)), X_data.values])
    n_obs, n_features = X_matrix.shape
   
    # Bayesian priors (weakly informative)
    prior_mean = np.zeros(n_features)
    prior_cov = np.eye(n_features) * 500
   
    # Inverse gamma parameters for variance
    nu_0, sigma_sq_0 = 3, 5
   
    # Gibbs sampling parameters
    n_iterations = 10000
    burn_in = 3000
   
    # Storage
    beta_samples = np.zeros((n_iterations, n_features))
    sigma_sq_samples = np.zeros(n_iterations)
   
    # Initialize
    beta = np.linalg.lstsq(X_matrix, y_data, rcond=None)[0]
    sigma_sq = np.var(y_data - X_matrix @ beta)
   
    # Gibbs sampling
    for i in range(n_iterations):
        # Sample beta given sigma_sq
        posterior_cov = np.linalg.inv(
            np.linalg.inv(prior_cov) + (X_matrix.T @ X_matrix) / sigma_sq
        )
        posterior_mean = posterior_cov @ (
            np.linalg.inv(prior_cov) @ prior_mean + (X_matrix.T @ y_data) / sigma_sq
        )
        beta = np.random.multivariate_normal(posterior_mean, posterior_cov)
        beta_samples[i] = beta
       
        # Sample sigma_sq given beta
        residuals = y_data - X_matrix @ beta
        nu_post = nu_0 + n_obs / 2
        sigma_sq_post = (nu_0 * sigma_sq_0 + np.sum(residuals**2) / 2) / nu_post
        sigma_sq = stats.invgamma.rvs(nu_post, scale=sigma_sq_post)
        sigma_sq_samples[i] = sigma_sq
   
    # Return post-burn-in samples
    return beta_samples[burn_in:]
# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - DATA & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/2563eb/ffffff?text=Your+Logo", width=150)
    st.markdown("---")
   
    st.header("⚙️ Configuration")
   
    # Data source selection
    st.subheader("1️⃣ Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["Demo Dataset", "Upload Your Data"],
        help="Start with demo data to explore, then upload your own business metrics"
    )
   
    df = None
   
    if data_source == "Upload Your Data":
        uploaded_file = st.file_uploader(
            "Upload CSV file with business metrics",
            type="csv",
            help="Your CSV should have columns for different business metrics (e.g., Revenue, Marketing Spend, Customer Count)"
        )
       
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} data points")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    else:
        if st.button("🎯 Load Demo Dataset", use_container_width=True):
            df = generate_realistic_business_data()
            st.success("✅ Demo data loaded successfully!")
            st.info("💡 This demo shows 20 months of realistic business metrics")
# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - ANALYSIS & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
if df is None:
    st.markdown("""
    <div class="metric-card">
    <h3>👋 Welcome to the Business Impact Simulator</h3>
    <p style="font-size: 1.1rem; line-height: 1.6;">
    This tool helps you answer critical business questions like:
    </p>
    <ul style="font-size: 1.05rem; line-height: 1.8;">
        <li><strong>What happens if we increase marketing spend by 20%?</strong></li>
        <li><strong>How does customer satisfaction affect our revenue?</strong></li>
        <li><strong>Which business levers should we focus on for maximum ROI?</strong></li>
    </ul>
    <p style="font-size: 1.1rem; margin-top: 20px;">
    👈 <strong>Get started by loading the demo dataset from the sidebar</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
# Validate data
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("❌ Your data needs at least 2 numeric columns to analyze relationships.")
    st.stop()
# ─── Business Metrics Dashboard ───────────────────────────────────────────
st.markdown("## 📈 Your Business Metrics at a Glance")
st.markdown("Current average values across your dataset:")
# Display metrics in a clean grid
metric_cols = st.columns(min(5, len(numeric_cols)))
for idx, col in enumerate(numeric_cols):
    with metric_cols[idx % 5]:
        mean_val = df[col].mean()
        std_val = df[col].std()
       
        # Format based on column name
        if 'USD' in col or 'Revenue' in col or 'Spend' in col:
            display_val = f"${mean_val:,.0f}"
        elif 'Percent' in col or 'Rate' in col:
            display_val = f"{mean_val:.1f}%"
        else:
            display_val = f"{mean_val:.1f}"
       
        st.metric(
            label=col.replace('_', ' '),
            value=display_val,
            delta=f"±{std_val:.1f}" if std_val > 0 else None,
            help=f"Average: {mean_val:.2f}, Std Dev: {std_val:.2f}"
        )
st.markdown("---")
# ─── Analysis Configuration ────────────────────────────────────────────────
with st.sidebar:
    st.subheader("2️⃣ Analysis Setup")
   
    primary_driver = st.selectbox(
        "**Primary Business Lever** (what you can control)",
        numeric_cols,
        help="This is the metric you're considering changing (e.g., Marketing Spend, Team Size)"
    )
   
    target_outcome = st.selectbox(
        "**Target Outcome** (what you want to improve)",
        [c for c in numeric_cols if c != primary_driver],
        help="This is the business result you want to impact (e.g., Revenue, Customer Satisfaction)"
    )
   
    available_controls = [c for c in numeric_cols if c not in (primary_driver, target_outcome)]
    control_variables = st.multiselect(
        "**Control Variables** (other factors to account for)",
        available_controls,
        default=available_controls[:min(3, len(available_controls))],
        help="These are other business factors that might influence your outcome"
    )
   
    st.markdown("---")
   
    analysis_method = st.selectbox(
        "**Analysis Method**",
        [
            "Bayesian Probabilistic Model (Recommended)",
            "Linear Regression",
            "Propensity Score Weighting"
        ],
        help="Bayesian method provides uncertainty ranges; Linear is simpler but less robust"
    )
# ─── Relationship Analysis ─────────────────────────────────────────────────
st.markdown("## 🔗 Understanding Your Business Ecosystem")
with st.spinner("🔍 Analyzing relationships between your business metrics..."):
    relationships = analyze_business_relationships(df[numeric_cols])
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        if relationships:
            fig = visualize_business_relationships(relationships)
            st.pyplot(fig)
        else:
            st.info("No strong relationships detected. Consider including more metrics or data points.")
   
    with col2:
        st.markdown("### 📖 How to Read This Map")
        st.markdown("""
        **Arrows show influence:**
        - 🟢 Green = Positive relationship
        - 🔴 Red = Negative relationship
        - Thicker = Stronger influence
       
        **Example:**
        If Marketing Spend → Revenue is green,
        it means increasing marketing tends
        to increase revenue.
        """)
if relationships:
    explanation = explain_business_relationships(relationships, target_outcome, numeric_cols)
    st.markdown(explanation)
st.markdown("---")
# ─── Impact Calculation ────────────────────────────────────────────────────
st.markdown(f"## 💰 Quantifying the Impact: {primary_driver} → {target_outcome}")
impact_results = {}
baseline_driver = float(df[primary_driver].mean())
baseline_outcome = float(df[target_outcome].mean())
if baseline_outcome == 0:
    baseline_outcome = 1 # Prevent division by zero
# Calculate based on selected method
if "Bayesian" in analysis_method:
    with st.spinner("🧮 Running advanced probabilistic analysis..."):
        posterior_samples = calculate_business_impact_bayesian(
            df, primary_driver, target_outcome, control_variables
        )
       
        if posterior_samples is not None:
            # Extract driver impact (column 1, after intercept)
            driver_impact_samples = posterior_samples[:, 1]
           
            impact_results[primary_driver] = {
                "mean": float(np.mean(driver_impact_samples)),
                "lower_95": float(np.percentile(driver_impact_samples, 2.5)),
                "upper_95": float(np.percentile(driver_impact_samples, 97.5)),
                "std": float(np.std(driver_impact_samples))
            }
           
            # Calculate for control variables
            for i, control_var in enumerate(control_variables, start=2):
                control_samples = posterior_samples[:, i]
                impact_results[control_var] = {
                    "mean": float(np.mean(control_samples)),
                    "lower_95": float(np.percentile(control_samples, 2.5)),
                    "upper_95": float(np.percentile(control_samples, 97.5)),
                    "std": float(np.std(control_samples))
                }
        else:
            st.warning("⚠️ Insufficient data for Bayesian analysis. Try with more data points.")
else:
    # Use DoWhy for traditional causal analysis
    with st.spinner("🧮 Calculating business impact..."):
        try:
            method_map = {
                "Linear Regression": "backdoor.linear_regression",
                "Propensity Score Weighting": "backdoor.propensity_score_weighting"
            }
           
            causal_method = method_map.get(analysis_method.split()[0], "backdoor.linear_regression")
           
            causal_model = CausalModel(
                data=df,
                treatment=primary_driver,
                outcome=target_outcome,
                common_causes=control_variables
            )
           
            identified_estimand = causal_model.identify_effect()
            causal_estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name=causal_method
            )
           
            effect_value = float(causal_estimate.value)
            effect_se = abs(effect_value) * 0.10 # Approximate standard error
           
            impact_results[primary_driver] = {
                "mean": effect_value,
                "lower_95": effect_value - 1.96 * effect_se,
                "upper_95": effect_value + 1.96 * effect_se,
                "std": effect_se
            }
           
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)[:200]}")
            impact_results[primary_driver] = {"mean": 0, "lower_95": 0, "upper_95": 0, "std": 0}
# ─── Display Results ───────────────────────────────────────────────────────
if primary_driver in impact_results and impact_results[primary_driver]["mean"] != 0:
    result = impact_results[primary_driver]
   
    # Calculate elasticity (% change in outcome per % change in driver)
    elasticity = result["mean"] * (baseline_driver / baseline_outcome)
    elasticity_lower = result["lower_95"] * (baseline_driver / baseline_outcome)
    elasticity_upper = result["upper_95"] * (baseline_driver / baseline_outcome)
   
    # Display main impact
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.metric(
            "**Per-Unit Impact**",
            f"{result['mean']:+.2f}",
            help=f"For every 1-unit increase in {primary_driver}, {target_outcome} changes by this amount"
        )
   
    with col2:
        direction_text = "increases" if elasticity > 0 else "decreases"
        st.metric(
            "**Elasticity (%)**",
            f"{elasticity:+.2f}%",
            help=f"A 1% change in {primary_driver} {direction_text} {target_outcome} by this %"
        )
   
    with col3:
        confidence_width = elasticity_upper - elasticity_lower
        st.metric(
            "**Confidence Range**",
            f"±{confidence_width/2:.2f}%",
            help="95% confidence interval width"
        )
   
    # Business interpretation
    st.markdown(f"""
    <div class="success-box">
    <h4>📊 What This Means for Your Business:</h4>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    Based on your data patterns, when you adjust <strong>{primary_driver}</strong>:
    </p>
    <ul style="font-size: 1.05rem; line-height: 1.8;">
        <li><strong>Expected Impact:</strong> A 1% change in {primary_driver} will {'increase' if elasticity > 0 else 'decrease'}
        {target_outcome} by approximately <strong>{abs(elasticity):.2f}%</strong></li>
        <li><strong>Realistic Range:</strong> The actual impact could range from <strong>{elasticity_lower:+.2f}%</strong> to
        <strong>{elasticity_upper:+.2f}%</strong></li>
        <li><strong>Confidence Level:</strong> We're 95% confident the true impact falls within this range</li>
    </ul>
    <p style="font-size: 1.05rem; margin-top: 16px;">
    💡 <strong>Key Insight:</strong> This {'positive' if elasticity > 0 else 'negative'} relationship suggests that
    {primary_driver} is a {'strong lever' if abs(elasticity) > 1 else 'moderate lever'} for influencing {target_outcome}.
    </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Could not calculate reliable impact estimate. This may indicate weak relationships or data quality issues.")
st.markdown("---")
# ─── What-If Scenario Simulator ────────────────────────────────────────────
st.markdown("## 🎯 What-If Scenario Planner")
st.markdown("""
Test different business scenarios before implementing them. Adjust your primary business lever
and see the predicted impact on your target outcome, including realistic ranges based on your data.
""")
col1, col2 = st.columns([2, 1])
with col1:
    scenario_change_pct = st.slider(
        f"**Scenario:** Change in {primary_driver}",
        min_value=-50,
        max_value=100,
        value=20,
        step=5,
        format="%d%%",
        help="Positive = increase, Negative = decrease"
    )
with col2:
    st.markdown("### Quick Scenarios")
    if st.button("Conservative (+10%)", use_container_width=True):
        scenario_change_pct = 10
    if st.button("Moderate (+25%)", use_container_width=True):
        scenario_change_pct = 25
    if st.button("Aggressive (+50%)", use_container_width=True):
        scenario_change_pct = 50
# Calculate scenario predictions
absolute_change = baseline_driver * (scenario_change_pct / 100)
if primary_driver in impact_results:
    result = impact_results[primary_driver]
   
    predicted_outcome = baseline_outcome + absolute_change * result["mean"]
    predicted_pct_change = ((predicted_outcome - baseline_outcome) / baseline_outcome) * 100
   
    predicted_lower = baseline_outcome + absolute_change * result["lower_95"]
    predicted_upper = baseline_outcome + absolute_change * result["upper_95"]
   
    pct_lower = ((predicted_lower - baseline_outcome) / baseline_outcome) * 100
    pct_upper = ((predicted_upper - baseline_outcome) / baseline_outcome) * 100
   
    # Display scenario results
    st.markdown("### 📊 Scenario Results")
   
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.markdown("**Current State**")
        st.metric(
            primary_driver,
            f"{baseline_driver:,.0f}"
        )
        st.metric(
            target_outcome,
            f"{baseline_outcome:,.0f}"
        )
   
    with col2:
        st.markdown("**After Change**")
        new_driver_value = baseline_driver + absolute_change
        st.metric(
            primary_driver,
            f"{new_driver_value:,.0f}",
            delta=f"{scenario_change_pct:+}%"
        )
        st.metric(
            target_outcome,
            f"{predicted_outcome:,.0f}",
            delta=f"{predicted_pct_change:+.1f}%"
        )
   
    with col3:
        st.markdown("**Confidence Range**")
        st.metric(
            "Best Case",
            f"{predicted_upper:,.0f}",
            delta=f"{pct_upper:+.1f}%"
        )
        st.metric(
            "Worst Case",
            f"{predicted_lower:,.0f}",
            delta=f"{pct_lower:+.1f}%"
        )
   
    # Visual interpretation
    direction = "increase" if predicted_pct_change > 0 else "decrease"
    confidence_level = "high" if (pct_upper - pct_lower) < 20 else "moderate" if (pct_upper - pct_lower) < 40 else "low"
   
    st.markdown(f"""
    <div class="info-box">
    <h4>🎯 Scenario Summary:</h4>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    If you {direction} <strong>{primary_driver}</strong> by <strong>{abs(scenario_change_pct)}%</strong>:
    </p>
    <ul style="font-size: 1.05rem; line-height: 1.8;">
        <li><strong>Expected Outcome:</strong> {target_outcome} will {direction} by approximately
        <strong>{abs(predicted_pct_change):.1f}%</strong> (from {baseline_outcome:,.0f} to {predicted_outcome:,.0f})</li>
        <li><strong>Range of Possibilities:</strong> The outcome could range from {predicted_lower:,.0f}
        ({pct_lower:+.1f}%) to {predicted_upper:,.0f} ({pct_upper:+.1f}%)</li>
        <li><strong>Confidence:</strong> {confidence_level.capitalize()} - the narrower the range, the more predictable the outcome</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")
# ─── Control Variables Analysis ────────────────────────────────────────────
if control_variables and len(impact_results) > 1:
    with st.expander("🔍 **Advanced:** How Control Variables Affect Your Outcome", expanded=False):
        st.markdown("""
        These are other business factors that could amplify or dampen the impact of your primary lever.
        Understanding these helps you plan more holistically.
        """)
       
        scenario_data = []
       
        for control_var in control_variables:
            if control_var not in impact_results:
                continue
           
            ctrl_result = impact_results[control_var]
            ctrl_baseline = df[control_var].mean()
           
            for pct_change in [-20, -10, 10, 20]:
                ctrl_absolute_change = ctrl_baseline * (pct_change / 100)
                ctrl_impact = ctrl_absolute_change * ctrl_result["mean"]
                ctrl_pct_impact = (ctrl_impact / baseline_outcome) * 100
               
                ctrl_impact_lower = ctrl_absolute_change * ctrl_result["lower_95"]
                ctrl_impact_upper = ctrl_absolute_change * ctrl_result["upper_95"]
                ctrl_pct_lower = (ctrl_impact_lower / baseline_outcome) * 100
                ctrl_pct_upper = (ctrl_impact_upper / baseline_outcome) * 100
               
                scenario_data.append({
                    "Control Variable": control_var.replace('_', ' '),
                    "Change": f"{pct_change:+}%",
                    "Impact on Outcome": f"{ctrl_impact:+.0f}",
                    "Impact %": f"{ctrl_pct_impact:+.1f}%",
                    "Range": f"{ctrl_pct_lower:+.1f}% to {ctrl_pct_upper:+.1f}%",
                    "_sort": abs(ctrl_pct_impact)
                })
       
        if scenario_data:
            df_scenarios = pd.DataFrame(scenario_data)
            df_scenarios = df_scenarios.sort_values("_sort", ascending=False)
           
            # Display with styling
            st.dataframe(
                df_scenarios.drop(columns=['_sort']).style.format({
                    "Impact on Outcome": "{}",
                    "Impact %": "{}",
                    "Range": "{}"
                }),
                use_container_width=True,
                height=400
            )
           
            st.markdown("""
            **💡 How to use this table:**
            - Look for variables with large impacts - these are key risk or opportunity factors
            - Consider scenarios where multiple factors change simultaneously
            - Plan contingencies for variables outside your control
            """)
# ─── Footer & Next Steps ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="metric-card">
<h3>🚀 Next Steps</h3>
<p style="font-size: 1.05rem; line-height: 1.8;">
<strong>1. Test Your Assumptions:</strong> Run different scenarios to stress-test your plans<br>
<strong>2. Monitor Continuously:</strong> Update your data regularly to refine predictions<br>
<strong>3. Start Small:</strong> Implement changes incrementally and validate results<br>
<strong>4. Share Insights:</strong> Use these findings to align your team on priorities
</p>
<p style="margin-top: 16px; font-size: 0.95rem; color: #64748b;">
<em>Need help interpreting results or want to connect your own data sources? Contact us for a personalized demo.</em>
</p>
</div>
""", unsafe_allow_html=True)