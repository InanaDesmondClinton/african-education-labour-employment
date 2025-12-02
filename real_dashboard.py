"""
African Education-Labour Analytics Dashboard
=============================================
A comprehensive, stakeholder-friendly dashboard for analyzing the relationship
between education indicators and labour market outcomes across Africa.

Author: Data Analytics Team
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="African Education-Labour Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563eb;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* DataFrame styling */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INDICATOR NAME MAPPINGS (HUMAN-READABLE)
# =============================================================================
INDICATOR_NAMES = {
    # Adjusted Net Attendance Rate (Primary Education)
    'SDG_NAT_AIR.1.GLAST': 'Primary School Attendance Rate (Overall)',
    'SDG_NAT_AIR.1.GLAST.F': 'Primary School Attendance Rate (Female)',
    'SDG_NAT_AIR.1.GLAST.M': 'Primary School Attendance Rate (Male)',
    'SDG_NAT_AIR.1.GLAST.GPIA': 'Primary School Gender Parity Index',
    
    # Adjusted Net Attendance Rate (Lower Secondary Education)
    'SDG_NAT_AIR.2.GPV.GLAST': 'Secondary School Attendance Rate (Overall)',
    'SDG_NAT_AIR.2.GPV.GLAST.F': 'Secondary School Attendance Rate (Female)',
    'SDG_NAT_AIR.2.GPV.GLAST.M': 'Secondary School Attendance Rate (Male)',
    'SDG_NAT_AIR.2.GPV.GLAST.GPIA': 'Secondary School Gender Parity Index',
    
    # Completion Rate
    'SDG_NAT_CR.MOD.1': 'Primary Education Completion Rate (Overall)',
    'SDG_NAT_CR.MOD.1.F': 'Primary Education Completion Rate (Female)',
    'SDG_NAT_CR.MOD.1.M': 'Primary Education Completion Rate (Male)',
    'SDG_NAT_CR.MOD.1.GPIA': 'Primary Completion Gender Parity Index',
    
    'SDG_NAT_CR.MOD.2': 'Lower Secondary Completion Rate (Overall)',
    'SDG_NAT_CR.MOD.2.F': 'Lower Secondary Completion Rate (Female)',
    'SDG_NAT_CR.MOD.2.M': 'Lower Secondary Completion Rate (Male)',
    'SDG_NAT_CR.MOD.2.GPIA': 'Lower Secondary Completion Gender Parity',
    
    'SDG_NAT_CR.MOD.3': 'Upper Secondary Completion Rate (Overall)',
    'SDG_NAT_CR.MOD.3.F': 'Upper Secondary Completion Rate (Female)',
    'SDG_NAT_CR.MOD.3.M': 'Upper Secondary Completion Rate (Male)',
    'SDG_NAT_CR.MOD.3.GPIA': 'Upper Secondary Completion Gender Parity',
    
    # Gross Enrollment Ratio
    'SDG_NAT_GER.5T8': 'Tertiary Education Enrollment (Gross)',
    
    # Out-of-School Rate
    'SDG_NAT_OAEPG.1': 'Out-of-School Rate (Overall)',
    'SDG_NAT_OAEPG.1.F': 'Out-of-School Rate (Female)',
    'SDG_NAT_OAEPG.1.M': 'Out-of-School Rate (Male)',
    'SDG_NAT_OAEPG.1.GPIA': 'Out-of-School Gender Parity Index',
    
    # Scholarships
    'SDG_NAT_ODAFLOW.VOLUMESCHOLARSHIP': 'International Scholarship Volume',
    
    # Trained Teachers
    'SDG_NAT_ROFST.1.CP': 'Trained Teachers in Primary Education (%)',
    'SDG_NAT_ROFST.MOD.1': 'Trained Teachers Primary (Overall)',
    'SDG_NAT_ROFST.MOD.1.F': 'Trained Teachers Primary (Female)',
    'SDG_NAT_ROFST.MOD.1.M': 'Trained Teachers Primary (Male)',
    'SDG_NAT_ROFST.MOD.1.GPIA': 'Trained Teachers Primary Gender Parity',
    
    'SDG_NAT_ROFST.MOD.2': 'Trained Teachers Lower Secondary (Overall)',
    'SDG_NAT_ROFST.MOD.2.F': 'Trained Teachers Lower Secondary (Female)',
    'SDG_NAT_ROFST.MOD.2.M': 'Trained Teachers Lower Secondary (Male)',
    'SDG_NAT_ROFST.MOD.2.GPIA': 'Trained Teachers Lower Secondary Gender Parity',
    
    'SDG_NAT_ROFST.MOD.3': 'Trained Teachers Upper Secondary (Overall)',
    'SDG_NAT_ROFST.MOD.3.F': 'Trained Teachers Upper Secondary (Female)',
    'SDG_NAT_ROFST.MOD.3.M': 'Trained Teachers Upper Secondary (Male)',
    'SDG_NAT_ROFST.MOD.3.GPIA': 'Trained Teachers Upper Secondary Gender Parity',
    
    # Government Expenditure
    'SDG_NAT_XGDP.FSGOV': 'Education Expenditure (% of GDP)',
    'SDG_NAT_XGOVEXP.IMF': 'Education Expenditure (% of Gov Budget)',
    
    # Years of Education
    'SDG_NAT_YEARS.FC.COMP.02': 'Years of Pre-primary Education (Compulsory)',
    'SDG_NAT_YEARS.FC.COMP.1T3': 'Years of Primary to Upper Secondary (Compulsory)',
    'SDG_NAT_YEARS.FC.FREE.02': 'Years of Free Pre-primary Education',
    'SDG_NAT_YEARS.FC.FREE.1T3': 'Years of Free Primary to Upper Secondary',
    
    # Labour indicators
    'Unemployment, youth total (% of total labor force ages 15-24)': 'Youth Unemployment Rate (15-24 years)',
    'Unemployment with advanced education (% of total labor force with advanced education)': 'Unemployment with Advanced Education',
    'Unemployment, total (% of total labor force)': 'Overall Unemployment Rate',
    
    # World Bank indicators
    'Foreign direct investment, net inflows (% of GDP)': 'Foreign Direct Investment (% GDP)',
    'Agriculture, forestry, and fishing, value added (% of GDP)': 'Agriculture Sector Contribution (% GDP)',
    'Industry (including construction), value added (% of GDP)': 'Industry Sector Contribution (% GDP)',
    'Services, value added (% of GDP)': 'Services Sector Contribution (% GDP)',
    'GDP growth (annual %)': 'Annual GDP Growth Rate',
    'Literacy rate, youth total (% of people ages 15-24)': 'Youth Literacy Rate (15-24 years)',
    'Literacy rate, adult total (% of people ages 15 and above)': 'Adult Literacy Rate (15+ years)',
    'Primary completion rate, total (% of relevant age group)': 'Primary Education Completion Rate',
    'Pupil-teacher ratio in primary education (headcount basis)': 'Student-Teacher Ratio (Primary)',
    'School enrollment, primary (% gross)': 'Primary School Enrollment (Gross %)',
    'School enrollment, secondary (% gross)': 'Secondary School Enrollment (Gross %)',
    'School enrollment, tertiary (% gross)': 'Tertiary Education Enrollment (Gross %)',
    'Government expenditure on education, total (% of government expenditure)': 'Education Expenditure (% of Gov Budget)',
    'Government expenditure on education, total (% of GDP)': 'Education Expenditure (% of GDP)',
    'Population ages 15-64 (% of total population)': 'Working-Age Population (15-64 years)',
    'Population growth (annual %)': 'Annual Population Growth Rate',
}

def get_readable_name(indicator):
    """Convert technical indicator code to human-readable name"""
    return INDICATOR_NAMES.get(indicator, indicator)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Load main datasets
        integrated_df = pd.read_csv('processed/integrated_data.csv')
        worldbank_df = pd.read_csv('processed/worldbank_cleaned.csv')
        sdg_national_df = pd.read_csv('processed/sdg_national_cleaned.csv')
        
        # Load metadata
        with open('processed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load analysis results
        with open('results/analysis_results.json', 'r') as f:
            analysis_results = json.load(f)
        
        return integrated_df, worldbank_df, sdg_national_df, metadata, analysis_results
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_metric_card(label, value, icon="📊"):
    """Create a custom metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def format_percentage(value):
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

def format_number(value):
    """Format number with comma separator"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_kpi_chart(data, title, color_scheme):
    """Create a KPI indicator chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = data['current'],
        delta = {'reference': data['previous'], 'relative': True},
        title = {"text": title},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_trend_chart(df, x_col, y_col, title, color='#667eea'):
    """Create a trend line chart"""
    fig = px.line(df, x=x_col, y=y_col, title=title,
                  labels={x_col: x_col.replace('_', ' ').title(),
                         y_col: get_readable_name(y_col)})
    
    fig.update_traces(line_color=color, line_width=3)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        title_font_size=16,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(corr_data, title):
    """Create correlation heatmap"""
    # Convert data to matrix format
    indicators = sorted(set([x['education_indicator'] for x in corr_data]))
    labour_indicators = sorted(set([x['labour_indicator'] for x in corr_data]))
    
    # Create matrix
    matrix = np.zeros((len(indicators), len(labour_indicators)))
    for item in corr_data:
        i = indicators.index(item['education_indicator'])
        j = labour_indicators.index(item['labour_indicator'])
        matrix[i, j] = item['correlation']
    
    # Readable names
    indicators_readable = [get_readable_name(x) for x in indicators]
    labour_readable = [get_readable_name(x) for x in labour_indicators]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labour_readable,
        y=indicators_readable,
        colorscale='RdBu',
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Labour Market Indicators",
        yaxis_title="Education Indicators",
        height=max(400, len(indicators) * 25),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_country_comparison(df, countries, indicator, year_range):
    """Create country comparison chart"""
    filtered_df = df[
        (df['country'].isin(countries)) & 
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    fig = px.line(filtered_df, x='year', y=indicator, color='country',
                  title=f'{get_readable_name(indicator)} - Country Comparison',
                  labels={'year': 'Year', indicator: get_readable_name(indicator)})
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_gender_gap_chart(df, indicator_base, countries, year):
    """Create gender gap analysis chart"""
    male_ind = f"{indicator_base}.M"
    female_ind = f"{indicator_base}.F"
    
    filtered_df = df[(df['year'] == year) & (df['country'].isin(countries))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Male',
        x=filtered_df['country'],
        y=filtered_df[male_ind],
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name='Female',
        x=filtered_df['country'],
        y=filtered_df[female_ind],
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        title=f'Gender Comparison: {get_readable_name(indicator_base)} ({year})',
        xaxis_title='Country',
        yaxis_title=get_readable_name(indicator_base),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Load data
    integrated_df, worldbank_df, sdg_national_df, metadata, analysis_results = load_data()
    
    if integrated_df is None:
        st.error("Failed to load data. Please ensure all data files are in the same directory.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🌍 African Education-Labour Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 Project Overview:</strong> This dashboard analyzes the relationship between educational 
        indicators and labour market outcomes across African countries, providing insights for policymakers, 
        educators, and stakeholders to bridge the gap between education outputs and employment demands.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Executive Dashboard", "📈 Data Exploration", "🔗 Correlation Analysis", 
         "🎯 Country Deep-Dive", "⚖️ Gender Gap Analysis", "📊 Statistical Insights",
         "🔮 Forecasting", "📥 Data Export"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        "**Data Sources:**\n"
        "- UNESCO SDG National Data\n"
        "- World Bank Open Data\n"
        "- ILO Statistics\n\n"
        "**Coverage:**\n"
        f"- Countries: {len(metadata['countries'])}\n"
        f"- Years: 2014-2021\n"
        f"- Indicators: 40+"
    )
    
    # Page routing
    if page == "🏠 Executive Dashboard":
        executive_dashboard(integrated_df, metadata, analysis_results)
    elif page == "📈 Data Exploration":
        data_exploration(integrated_df, worldbank_df, sdg_national_df)
    elif page == "🔗 Correlation Analysis":
        correlation_analysis(integrated_df, analysis_results)
    elif page == "🎯 Country Deep-Dive":
        country_deep_dive(integrated_df, metadata)
    elif page == "⚖️ Gender Gap Analysis":
        gender_gap_analysis(integrated_df, metadata)
    elif page == "📊 Statistical Insights":
        statistical_insights(analysis_results)
    elif page == "🔮 Forecasting":
        forecasting_page(integrated_df)
    elif page == "📥 Data Export":
        data_export(integrated_df, worldbank_df, sdg_national_df)

# =============================================================================
# PAGE: EXECUTIVE DASHBOARD
# =============================================================================
def executive_dashboard(df, metadata, analysis_results):
    """Main executive dashboard with key metrics and insights"""
    
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Key Performance Indicators
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_primary_completion = df['SDG_NAT_CR.MOD.1'].mean()
        st.markdown(create_metric_card(
            "Avg Primary Completion Rate", 
            f"{avg_primary_completion:.1f}%",
            "🎓"
        ), unsafe_allow_html=True)
    
    with col2:
        avg_youth_unemployment = df['Unemployment, youth total (% of total labor force ages 15-24)'].mean()
        st.markdown(create_metric_card(
            "Avg Youth Unemployment", 
            f"{avg_youth_unemployment:.1f}%",
            "💼"
        ), unsafe_allow_html=True)
    
    with col3:
        avg_edu_expenditure = df['SDG_NAT_XGDP.FSGOV'].mean()
        st.markdown(create_metric_card(
            "Avg Education Spending", 
            f"{avg_edu_expenditure:.1f}% GDP",
            "💰"
        ), unsafe_allow_html=True)
    
    with col4:
        countries_covered = len(metadata['countries'])
        st.markdown(create_metric_card(
            "Countries Analyzed", 
            f"{countries_covered}",
            "🌍"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Continental Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Education Landscape Overview")
        
        # Calculate averages by year
        edu_trends = df.groupby('year').agg({
            'SDG_NAT_CR.MOD.1': 'mean',  # Primary completion
            'SDG_NAT_CR.MOD.2': 'mean',  # Lower secondary completion
            'SDG_NAT_CR.MOD.3': 'mean',  # Upper secondary completion
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edu_trends['year'], y=edu_trends['SDG_NAT_CR.MOD.1'],
            name='Primary', mode='lines+markers', line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=edu_trends['year'], y=edu_trends['SDG_NAT_CR.MOD.2'],
            name='Lower Secondary', mode='lines+markers', line=dict(color='#764ba2', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=edu_trends['year'], y=edu_trends['SDG_NAT_CR.MOD.3'],
            name='Upper Secondary', mode='lines+markers', line=dict(color='#f59e0b', width=3)
        ))
        
        fig.update_layout(
            title='Education Completion Rates Trend (2014-2021)',
            xaxis_title='Year',
            yaxis_title='Completion Rate (%)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💼 Labour Market Overview")
        
        # Calculate unemployment trends
        labour_trends = df.groupby('year').agg({
            'Unemployment, total (% of total labor force)': 'mean',
            'Unemployment, youth total (% of total labor force ages 15-24)': 'mean',
            'Unemployment with advanced education (% of total labor force with advanced education)': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labour_trends['year'], 
            y=labour_trends['Unemployment, total (% of total labor force)'],
            name='Overall Unemployment', mode='lines+markers', line=dict(color='#ef4444', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=labour_trends['year'], 
            y=labour_trends['Unemployment, youth total (% of total labor force ages 15-24)'],
            name='Youth Unemployment', mode='lines+markers', line=dict(color='#f59e0b', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=labour_trends['year'], 
            y=labour_trends['Unemployment with advanced education (% of total labor force with advanced education)'],
            name='Advanced Education', mode='lines+markers', line=dict(color='#8b5cf6', width=3)
        ))
        
        fig.update_layout(
            title='Unemployment Rates Trend (2014-2021)',
            xaxis_title='Year',
            yaxis_title='Unemployment Rate (%)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Regional Performance
    st.markdown("### 🗺️ Top & Bottom Performing Countries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Highest Primary Completion Rates (2021)")
        latest_year = df['year'].max()
        top_completion = df[df['year'] == latest_year].nlargest(10, 'SDG_NAT_CR.MOD.1')[
            ['country', 'SDG_NAT_CR.MOD.1']
        ].reset_index(drop=True)
        top_completion.columns = ['Country', 'Completion Rate (%)']
        top_completion.index = top_completion.index + 1
        st.dataframe(top_completion, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ Lowest Primary Completion Rates (2021)")
        bottom_completion = df[df['year'] == latest_year].nsmallest(10, 'SDG_NAT_CR.MOD.1')[
            ['country', 'SDG_NAT_CR.MOD.1']
        ].reset_index(drop=True)
        bottom_completion.columns = ['Country', 'Completion Rate (%)']
        bottom_completion.index = bottom_completion.index + 1
        st.dataframe(bottom_completion, use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 💡 Key Insights & Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Positive Trends</h4>
            <ul>
                <li>Primary completion rates showing steady improvement</li>
                <li>Gender parity in education improving across the continent</li>
                <li>Increased government expenditure on education</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Areas of Concern</h4>
            <ul>
                <li>Youth unemployment remains critically high</li>
                <li>Skills mismatch between education and job market</li>
                <li>Secondary and tertiary completion rates lag behind</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Key Correlations</h4>
            <ul>
                <li>Primary completion strongly correlates with youth unemployment</li>
                <li>Education spending impacts long-term outcomes</li>
                <li>Gender parity influences overall development</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Geographic Distribution
    st.markdown("---")
    st.markdown("### 🌍 Geographic Distribution")
    
    # Create map visualization
    latest_data = df[df['year'] == df['year'].max()].copy()
    
    fig = px.scatter_geo(
        latest_data,
        locations="country_id",
        locationmode='ISO-3',
        color="SDG_NAT_CR.MOD.1",
        hover_name="country",
        hover_data={
            'SDG_NAT_CR.MOD.1': ':.2f',
            'Unemployment, youth total (% of total labor force ages 15-24)': ':.2f',
            'SDG_NAT_XGDP.FSGOV': ':.2f'
        },
        size="SDG_NAT_XGDP.FSGOV",
        title="Primary Completion Rates Across Africa (Latest Year)",
        color_continuous_scale='Viridis',
        labels={'SDG_NAT_CR.MOD.1': 'Completion Rate (%)'}
    )
    
    fig.update_geos(
        scope='africa',
        showcountries=True,
        countrycolor="lightgray"
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: DATA EXPLORATION
# =============================================================================
def data_exploration(integrated_df, worldbank_df, sdg_national_df):
    """Interactive data exploration page"""
    
    st.markdown('<h2 class="sub-header">📈 Data Exploration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🔍 Filter & Explore", "📉 Distribution Analysis"])
    
    with tab1:
        st.markdown("### 📦 Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Integrated Dataset**\n\n"
                   f"- Records: {len(integrated_df):,}\n"
                   f"- Columns: {len(integrated_df.columns)}\n"
                   f"- Countries: {integrated_df['country'].nunique()}\n"
                   f"- Years: {integrated_df['year'].min()}-{integrated_df['year'].max()}")
        
        with col2:
            st.info(f"**World Bank Data**\n\n"
                   f"- Records: {len(worldbank_df):,}\n"
                   f"- Columns: {len(worldbank_df.columns)}\n"
                   f"- Countries: {worldbank_df['country'].nunique()}\n"
                   f"- Years: {worldbank_df['year'].min()}-{worldbank_df['year'].max()}")
        
        with col3:
            st.info(f"**SDG National Data**\n\n"
                   f"- Records: {len(sdg_national_df):,}\n"
                   f"- Columns: {len(sdg_national_df.columns)}\n"
                   f"- Countries: {sdg_national_df['country'].nunique()}\n"
                   f"- Years: {sdg_national_df['year'].min()}-{sdg_national_df['year'].max()}")
        
        st.markdown("---")
        st.markdown("### 📋 Column Overview")
        
        # Display column information
        col_info = pd.DataFrame({
            'Column': integrated_df.columns,
            'Type': integrated_df.dtypes,
            'Missing (%)': (integrated_df.isnull().sum() / len(integrated_df) * 100).round(2),
            'Unique': integrated_df.nunique()
        })
        
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔍 Interactive Data Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_countries = st.multiselect(
                "Select Countries:",
                options=sorted(integrated_df['country'].unique()),
                default=list(sorted(integrated_df['country'].unique())[:5])
            )
        
        with col2:
            year_range = st.slider(
                "Select Year Range:",
                min_value=int(integrated_df['year'].min()),
                max_value=int(integrated_df['year'].max()),
                value=(int(integrated_df['year'].min()), int(integrated_df['year'].max()))
            )
        
        with col3:
            # Get numeric columns only
            numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
            # Remove year and country_id
            indicator_cols = [col for col in numeric_cols if col not in ['year', 'country_id']]
            
            selected_indicator = st.selectbox(
                "Select Indicator:",
                options=indicator_cols,
                format_func=lambda x: get_readable_name(x)
            )
        
        # Filter data
        filtered_df = integrated_df[
            (integrated_df['country'].isin(selected_countries)) &
            (integrated_df['year'] >= year_range[0]) &
            (integrated_df['year'] <= year_range[1])
        ]
        
        if len(filtered_df) > 0:
            st.markdown(f"**Filtered Data: {len(filtered_df)} records**")
            
            # Visualization
            fig = px.line(
                filtered_df,
                x='year',
                y=selected_indicator,
                color='country',
                title=f'{get_readable_name(selected_indicator)} Over Time',
                labels={'year': 'Year', selected_indicator: get_readable_name(selected_indicator)}
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("### 📄 Filtered Data Table")
            display_cols = ['country', 'year', selected_indicator]
            st.dataframe(filtered_df[display_cols].sort_values(['country', 'year']), 
                        use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab3:
        st.markdown("### 📉 Distribution Analysis")
        
        # Select indicator for distribution
        numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
        indicator_cols = [col for col in numeric_cols if col not in ['year', 'country_id']]
        
        selected_dist_indicator = st.selectbox(
            "Select Indicator for Distribution:",
            options=indicator_cols,
            format_func=lambda x: get_readable_name(x),
            key='dist_indicator'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                integrated_df,
                x=selected_dist_indicator,
                nbins=30,
                title=f'Distribution of {get_readable_name(selected_dist_indicator)}',
                labels={selected_dist_indicator: get_readable_name(selected_dist_indicator)}
            )
            
            fig.update_traces(marker_color='#667eea')
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by year
            fig = px.box(
                integrated_df,
                x='year',
                y=selected_dist_indicator,
                title=f'{get_readable_name(selected_dist_indicator)} by Year',
                labels={'year': 'Year', selected_dist_indicator: get_readable_name(selected_dist_indicator)}
            )
            
            fig.update_traces(marker_color='#764ba2')
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 Statistical Summary")
        
        stats_df = integrated_df[selected_dist_indicator].describe().to_frame()
        stats_df.columns = [get_readable_name(selected_dist_indicator)]
        stats_df = stats_df.T
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{stats_df['mean'].values[0]:.2f}")
        with col2:
            st.metric("Median", f"{stats_df['50%'].values[0]:.2f}")
        with col3:
            st.metric("Std Dev", f"{stats_df['std'].values[0]:.2f}")
        with col4:
            st.metric("Range", f"{stats_df['max'].values[0] - stats_df['min'].values[0]:.2f}")

# =============================================================================
# PAGE: CORRELATION ANALYSIS
# =============================================================================
def correlation_analysis(df, analysis_results):
    """Correlation analysis page"""
    
    st.markdown('<h2 class="sub-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This section explores the relationships between education indicators and labour market outcomes.
        Strong correlations help identify which educational factors most influence employment.
    </div>
    """, unsafe_allow_html=True)
    
    # Get correlation data
    if 'correlations' in analysis_results:
        corr_data = analysis_results['correlations']
        
        # Filter for significant correlations (top correlations)
        significant_corr = [c for c in corr_data if abs(c['correlation']) > 0.3]
        significant_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Top correlations
        st.markdown("### 🔝 Top Education-Labour Correlations")
        
        top_10 = significant_corr[:10]
        
        # Create dataframe for display
        corr_display = pd.DataFrame([{
            'Education Indicator': get_readable_name(c['education_indicator']),
            'Labour Indicator': get_readable_name(c['labour_indicator']),
            'Correlation': f"{c['correlation']:.3f}",
            'Significance': c['significance'],
            'Sample Size': c['sample_size']
        } for c in top_10])
        
        st.dataframe(corr_display, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of top correlations
            fig = go.Figure()
            
            colors = ['#667eea' if c['correlation'] > 0 else '#ef4444' for c in top_10]
            
            fig.add_trace(go.Bar(
                x=[c['correlation'] for c in top_10],
                y=[f"{get_readable_name(c['education_indicator'][:30])}<br>→ {get_readable_name(c['labour_indicator'][:30])}" 
                   for c in top_10],
                orientation='h',
                marker_color=colors,
                text=[f"{c['correlation']:.3f}" for c in top_10],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Top 10 Correlations (Education → Labour)',
                xaxis_title='Correlation Coefficient',
                yaxis_title='',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot for selected correlation
            st.markdown("#### 🔍 Detailed Correlation View")
            
            selected_corr = st.selectbox(
                "Select correlation to explore:",
                options=range(len(top_10)),
                format_func=lambda x: f"{get_readable_name(top_10[x]['education_indicator'])} → {get_readable_name(top_10[x]['labour_indicator'])}"
            )
            
            edu_ind = top_10[selected_corr]['education_indicator']
            lab_ind = top_10[selected_corr]['labour_indicator']
            corr_val = top_10[selected_corr]['correlation']
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x=edu_ind,
                y=lab_ind,
                trendline='ols',
                title=f'Correlation: {corr_val:.3f}',
                labels={
                    edu_ind: get_readable_name(edu_ind),
                    lab_ind: get_readable_name(lab_ind)
                },
                opacity=0.6
            )
            
            fig.update_traces(marker=dict(size=8, color='#667eea'))
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.markdown("---")
        st.markdown("### 🗺️ Correlation Heatmap")
        
        # Select specific indicators for heatmap
        education_indicators = list(set([c['education_indicator'] for c in corr_data]))
        labour_indicators = list(set([c['labour_indicator'] for c in corr_data]))
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_edu_indicators = st.multiselect(
                "Select Education Indicators:",
                options=education_indicators,
                default=education_indicators[:10],
                format_func=lambda x: get_readable_name(x)
            )
        
        with col2:
            selected_lab_indicators = st.multiselect(
                "Select Labour Indicators:",
                options=labour_indicators,
                default=labour_indicators,
                format_func=lambda x: get_readable_name(x)
            )
        
        if selected_edu_indicators and selected_lab_indicators:
            # Filter correlation data
            filtered_corr = [c for c in corr_data 
                           if c['education_indicator'] in selected_edu_indicators 
                           and c['labour_indicator'] in selected_lab_indicators]
            
            if filtered_corr:
                fig = create_correlation_heatmap(filtered_corr, 
                    "Education-Labour Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        st.markdown("---")
        st.markdown("### 📖 Interpretation Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>Strong Positive (0.5 to 1.0)</h4>
                <p>As education indicator increases, unemployment increases.
                This may indicate education-job mismatch.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>Moderate (0.3 to 0.5)</h4>
                <p>Some relationship exists but other factors also play significant roles.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>Weak (< 0.3)</h4>
                <p>Little to no linear relationship. May have non-linear relationships.</p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE: COUNTRY DEEP-DIVE
# =============================================================================
def country_deep_dive(df, metadata):
    """Country-specific analysis page"""
    
    st.markdown('<h2 class="sub-header">🎯 Country Deep-Dive Analysis</h2>', unsafe_allow_html=True)
    
    # Country selection
    selected_country = st.selectbox(
        "Select Country:",
        options=sorted(metadata['countries']),
        index=sorted(metadata['countries']).index('Nigeria') if 'Nigeria' in metadata['countries'] else 0
    )
    
    # Filter data for selected country
    country_data = df[df['country'] == selected_country].sort_values('year')
    
    if len(country_data) == 0:
        st.warning(f"No data available for {selected_country}")
        return
    
    # Overview metrics
    st.markdown(f"### 📊 {selected_country} - Key Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = country_data.iloc[-1]
    
    with col1:
        val = latest_data['SDG_NAT_CR.MOD.1']
        st.metric("Primary Completion", f"{val:.1f}%")
    
    with col2:
        val = latest_data['Unemployment, youth total (% of total labor force ages 15-24)']
        st.metric("Youth Unemployment", f"{val:.1f}%")
    
    with col3:
        val = latest_data['SDG_NAT_XGDP.FSGOV']
        st.metric("Education Spending", f"{val:.1f}% GDP")
    
    with col4:
        val = latest_data['Literacy rate, adult total (% of people ages 15 and above)']
        st.metric("Adult Literacy", f"{val:.1f}%")
    
    st.markdown("---")
    
    # Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Education Trends")
        
        fig = go.Figure()
        
        indicators = [
            ('SDG_NAT_CR.MOD.1', 'Primary Completion', '#667eea'),
            ('SDG_NAT_CR.MOD.2', 'Lower Secondary Completion', '#764ba2'),
            ('SDG_NAT_CR.MOD.3', 'Upper Secondary Completion', '#f59e0b')
        ]
        
        for ind, name, color in indicators:
            if ind in country_data.columns:
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data[ind],
                    name=name,
                    mode='lines+markers',
                    line=dict(color=color, width=3)
                ))
        
        fig.update_layout(
            title=f'Education Completion Rates - {selected_country}',
            xaxis_title='Year',
            yaxis_title='Completion Rate (%)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💼 Labour Market Trends")
        
        fig = go.Figure()
        
        indicators = [
            ('Unemployment, total (% of total labor force)', 'Overall Unemployment', '#ef4444'),
            ('Unemployment, youth total (% of total labor force ages 15-24)', 'Youth Unemployment', '#f59e0b')
        ]
        
        for ind, name, color in indicators:
            if ind in country_data.columns:
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data[ind],
                    name=name,
                    mode='lines+markers',
                    line=dict(color=color, width=3)
                ))
        
        fig.update_layout(
            title=f'Unemployment Rates - {selected_country}',
            xaxis_title='Year',
            yaxis_title='Unemployment Rate (%)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Economic indicators
    st.markdown("#### 💰 Economic Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['GDP growth (annual %)'],
            name='GDP Growth',
            mode='lines+markers',
            line=dict(color='#22c55e', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f'GDP Growth Rate - {selected_country}',
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector contribution
        latest_year = country_data['year'].max()
        latest_row = country_data[country_data['year'] == latest_year].iloc[0]
        
        sectors = {
            'Agriculture': latest_row['Agriculture, forestry, and fishing, value added (% of GDP)'],
            'Industry': latest_row['Industry (including construction), value added (% of GDP)'],
            'Services': latest_row['Services, value added (% of GDP)']
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(sectors.keys()),
            values=list(sectors.values()),
            hole=.3,
            marker_colors=['#667eea', '#764ba2', '#f59e0b']
        )])
        
        fig.update_layout(
            title=f'GDP Composition by Sector ({latest_year}) - {selected_country}',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed data table
    st.markdown("#### 📄 Detailed Country Data")
    
    # Select relevant columns
    display_cols = [
        'year',
        'SDG_NAT_CR.MOD.1',
        'SDG_NAT_CR.MOD.2',
        'SDG_NAT_CR.MOD.3',
        'Unemployment, youth total (% of total labor force ages 15-24)',
        'Unemployment, total (% of total labor force)',
        'SDG_NAT_XGDP.FSGOV',
        'GDP growth (annual %)'
    ]
    
    # Filter columns that exist
    display_cols = [col for col in display_cols if col in country_data.columns]
    
    # Rename columns for display
    display_df = country_data[display_cols].copy()
    display_df.columns = [get_readable_name(col) if col != 'year' else 'Year' for col in display_cols]
    
    st.dataframe(display_df.sort_values('Year', ascending=False), use_container_width=True)

# =============================================================================
# PAGE: GENDER GAP ANALYSIS
# =============================================================================
def gender_gap_analysis(df, metadata):
    """Gender gap analysis page"""
    
    st.markdown('<h2 class="sub-header">⚖️ Gender Gap Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Analyzing gender disparities in education across Africa. Gender Parity Index (GPI) values:
        - <strong>GPI = 1</strong>: Perfect gender parity
        - <strong>GPI < 1</strong>: Disparity in favor of males
        - <strong>GPI > 1</strong>: Disparity in favor of females
    </div>
    """, unsafe_allow_html=True)
    
    # Year selection
    selected_year = st.select_slider(
        "Select Year:",
        options=sorted(df['year'].unique()),
        value=df['year'].max()
    )
    
    year_data = df[df['year'] == selected_year]
    
    # Gender Parity Overview
    st.markdown("### 🎯 Gender Parity Index Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_gpi_primary = year_data['SDG_NAT_CR.MOD.1.GPIA'].mean()
        st.metric(
            "Primary Completion GPI",
            f"{avg_gpi_primary:.3f}",
            delta=f"{avg_gpi_primary - 1:.3f} from parity"
        )
    
    with col2:
        avg_gpi_secondary = year_data['SDG_NAT_CR.MOD.2.GPIA'].mean()
        st.metric(
            "Lower Secondary GPI",
            f"{avg_gpi_secondary:.3f}",
            delta=f"{avg_gpi_secondary - 1:.3f} from parity"
        )
    
    with col3:
        avg_gpi_upper = year_data['SDG_NAT_CR.MOD.3.GPIA'].mean()
        st.metric(
            "Upper Secondary GPI",
            f"{avg_gpi_upper:.3f}",
            delta=f"{avg_gpi_upper - 1:.3f} from parity"
        )
    
    st.markdown("---")
    
    # Male vs Female comparison
    st.markdown("### 👥 Male vs Female Completion Rates")
    
    # Country selection
    selected_countries = st.multiselect(
        "Select Countries for Comparison:",
        options=sorted(metadata['countries']),
        default=list(sorted(metadata['countries'])[:10])
    )
    
    if selected_countries:
        filtered_data = year_data[year_data['country'].isin(selected_countries)]
        
        # Primary education comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Male',
            x=filtered_data['country'],
            y=filtered_data['SDG_NAT_CR.MOD.1.M'],
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            name='Female',
            x=filtered_data['country'],
            y=filtered_data['SDG_NAT_CR.MOD.1.F'],
            marker_color='#764ba2'
        ))
        
        fig.update_layout(
            title=f'Primary Completion Rate by Gender ({selected_year})',
            xaxis_title='Country',
            yaxis_title='Completion Rate (%)',
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # GPI Map
        st.markdown("### 🗺️ Gender Parity Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # GPI scatter plot
            fig = px.scatter(
                filtered_data,
                x='SDG_NAT_CR.MOD.1.M',
                y='SDG_NAT_CR.MOD.1.F',
                text='country',
                title='Male vs Female Primary Completion',
                labels={
                    'SDG_NAT_CR.MOD.1.M': 'Male Completion Rate (%)',
                    'SDG_NAT_CR.MOD.1.F': 'Female Completion Rate (%)'
                },
                color='SDG_NAT_CR.MOD.1.GPIA',
                color_continuous_scale='RdYlGn',
                size='SDG_NAT_CR.MOD.1.GPIA'
            )
            
            # Add diagonal line for parity
            max_val = max(filtered_data['SDG_NAT_CR.MOD.1.M'].max(), 
                         filtered_data['SDG_NAT_CR.MOD.1.F'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Parity',
                showlegend=True
            ))
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # GPI bar chart
            fig = go.Figure()
            
            # Sort by GPI
            sorted_data = filtered_data.sort_values('SDG_NAT_CR.MOD.1.GPIA', ascending=False)
            
            # Color bars based on GPI value
            colors = ['#22c55e' if gpi >= 0.95 and gpi <= 1.05 else '#f59e0b' if gpi >= 0.9 and gpi <= 1.1 else '#ef4444' 
                     for gpi in sorted_data['SDG_NAT_CR.MOD.1.GPIA']]
            
            fig.add_trace(go.Bar(
                x=sorted_data['country'],
                y=sorted_data['SDG_NAT_CR.MOD.1.GPIA'],
                marker_color=colors,
                text=sorted_data['SDG_NAT_CR.MOD.1.GPIA'].round(3),
                textposition='outside'
            ))
            
            # Add parity line
            fig.add_hline(y=1, line_dash="dash", line_color="red", 
                         annotation_text="Perfect Parity (1.0)")
            
            fig.update_layout(
                title=f'Gender Parity Index - Primary Education ({selected_year})',
                xaxis_title='Country',
                yaxis_title='Gender Parity Index',
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Trends over time
        st.markdown("### 📈 Gender Parity Trends Over Time")
        
        selected_country_trend = st.selectbox(
            "Select Country for Trend Analysis:",
            options=sorted(metadata['countries'])
        )
        
        country_trend_data = df[df['country'] == selected_country_trend].sort_values('year')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=country_trend_data['year'],
            y=country_trend_data['SDG_NAT_CR.MOD.1.GPIA'],
            name='Primary GPI',
            mode='lines+markers',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=country_trend_data['year'],
            y=country_trend_data['SDG_NAT_CR.MOD.2.GPIA'],
            name='Lower Secondary GPI',
            mode='lines+markers',
            line=dict(color='#764ba2', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=country_trend_data['year'],
            y=country_trend_data['SDG_NAT_CR.MOD.3.GPIA'],
            name='Upper Secondary GPI',
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3)
        ))
        
        # Add parity line
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Perfect Parity")
        
        fig.update_layout(
            title=f'Gender Parity Index Trends - {selected_country_trend}',
            xaxis_title='Year',
            yaxis_title='Gender Parity Index',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: STATISTICAL INSIGHTS
# =============================================================================
def statistical_insights(analysis_results):
    """Statistical insights from analysis"""
    
    st.markdown('<h2 class="sub-header">📊 Statistical Insights</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Statistical analysis results from the integrated dataset, including distribution statistics,
        outlier analysis, and data quality metrics.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Distribution Statistics", "⚠️ Outlier Analysis", "✅ Data Quality"])
    
    with tab1:
        st.markdown("### 📊 Distribution Statistics")
        
        if 'eda' in analysis_results and 'distribution_stats' in analysis_results['eda']:
            stats = analysis_results['eda']['distribution_stats']
            
            # Convert to dataframe
            stats_df = pd.DataFrame(stats).T
            stats_df.index = [get_readable_name(idx) for idx in stats_df.index]
            stats_df = stats_df.round(3)
            
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 📉 Skewness and Kurtosis Interpretation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>Skewness</h4>
                    <ul>
                        <li><strong>-0.5 to 0.5:</strong> Fairly symmetrical</li>
                        <li><strong>0.5 to 1.0:</strong> Moderately skewed</li>
                        <li><strong>> 1.0:</strong> Highly skewed</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <h4>Kurtosis</h4>
                    <ul>
                        <li><strong>~0:</strong> Normal distribution (mesokurtic)</li>
                        <li><strong>> 0:</strong> Heavy tails (leptokurtic)</li>
                        <li><strong>< 0:</strong> Light tails (platykurtic)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### ⚠️ Outlier Analysis")
        
        if 'eda' in analysis_results and 'outlier_summary' in analysis_results['eda']:
            outliers = analysis_results['eda']['outlier_summary']
            
            # Convert to dataframe
            outlier_df = pd.DataFrame(outliers).T
            outlier_df.index = [get_readable_name(idx) for idx in outlier_df.index]
            outlier_df = outlier_df.round(3)
            
            st.markdown("#### Outlier Detection Summary (IQR Method)")
            st.dataframe(outlier_df, use_container_width=True)
            
            st.markdown("---")
            
            # Visualize outlier percentages
            outlier_pcts = {get_readable_name(k): v['outlier_pct'] 
                          for k, v in outliers.items()}
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(outlier_pcts.keys()),
                y=list(outlier_pcts.values()),
                marker_color='#ef4444',
                text=[f"{v:.1f}%" for v in outlier_pcts.values()],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Outlier Percentage by Indicator',
                xaxis_title='Indicator',
                yaxis_title='Outlier Percentage (%)',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ✅ Data Quality Metrics")
        
        if 'sdg_analysis' in analysis_results and 'availability' in analysis_results['sdg_analysis']:
            availability = analysis_results['sdg_analysis']['availability']
            
            # Calculate overall completeness
            total_records = sum([v['count'] for v in availability.values()])
            total_possible = len(availability) * list(availability.values())[0]['count'] if availability else 0
            
            completeness = (total_records / total_possible * 100) if total_possible > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Data Completeness", f"{completeness:.1f}%")
            with col2:
                st.metric("Total Indicators", len(availability))
            with col3:
                avg_coverage = np.mean([v['percentage'] for v in availability.values()])
                st.metric("Average Coverage", f"{avg_coverage:.1f}%")
            
            st.markdown("---")
            
            # Availability table
            avail_df = pd.DataFrame(availability).T
            avail_df.index = [get_readable_name(idx) for idx in avail_df.index]
            avail_df = avail_df.sort_values('percentage', ascending=False)
            
            st.markdown("#### Indicator Data Availability")
            st.dataframe(avail_df, use_container_width=True)

# =============================================================================
# PAGE: FORECASTING
# =============================================================================
def forecasting_page(df):
    """Forecasting and prediction page"""
    
    st.markdown('<h2 class="sub-header">🔮 Time Series Forecasting</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Note:</strong> This is a simplified forecasting demonstration. For production use,
        implement proper time series models (ARIMA, Prophet, LSTM) with train-test validation.
    </div>
    """, unsafe_allow_html=True)
    
    # Country and indicator selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_country = st.selectbox(
            "Select Country:",
            options=sorted(df['country'].unique())
        )
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicator_cols = [col for col in numeric_cols if col not in ['year', 'country_id']]
        
        selected_indicator = st.selectbox(
            "Select Indicator to Forecast:",
            options=indicator_cols,
            format_func=lambda x: get_readable_name(x)
        )
    
    # Filter data
    country_data = df[df['country'] == selected_country].sort_values('year')
    
    if len(country_data) < 3:
        st.warning("Insufficient data for forecasting.")
        return
    
    # Simple linear extrapolation (for demonstration)
    years = country_data['year'].values
    values = country_data[selected_indicator].values
    
    # Remove NaN values
    mask = ~np.isnan(values)
    years = years[mask]
    values = values[mask]
    
    if len(years) < 2:
        st.warning("Insufficient non-null data for forecasting.")
        return
    
    # Fit linear trend
    z = np.polyfit(years, values, 1)
    p = np.poly1d(z)
    
    # Forecast future years
    forecast_years = np.arange(years[-1] + 1, years[-1] + 6)
    forecast_values = p(forecast_years)
    
    # Visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#667eea', width=3)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#f59e0b', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{get_readable_name(selected_indicator)} Forecast - {selected_country}',
        xaxis_title='Year',
        yaxis_title=get_readable_name(selected_indicator),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("### 📋 Forecast Values")
    
    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        get_readable_name(selected_indicator): forecast_values.round(2)
    })
    
    st.dataframe(forecast_df, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Disclaimer:</strong> This forecast uses simple linear extrapolation for demonstration purposes.
        Real-world forecasting should account for seasonality, trends, external factors, and use validated models.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: DATA EXPORT
# =============================================================================
def data_export(integrated_df, worldbank_df, sdg_national_df):
    """Data export and download page"""
    
    st.markdown('<h2 class="sub-header">📥 Data Export</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Download filtered data for offline analysis or reporting.
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    dataset_choice = st.selectbox(
        "Select Dataset:",
        options=["Integrated Dataset", "World Bank Data", "SDG National Data"]
    )
    
    if dataset_choice == "Integrated Dataset":
        working_df = integrated_df
    elif dataset_choice == "World Bank Data":
        working_df = worldbank_df
    else:
        working_df = sdg_national_df
    
    st.markdown(f"**Dataset Size:** {len(working_df):,} records × {len(working_df.columns)} columns")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'country' in working_df.columns:
            selected_countries = st.multiselect(
                "Filter by Countries:",
                options=sorted(working_df['country'].unique()),
                default=[]
            )
        else:
            selected_countries = []
    
    with col2:
        if 'year' in working_df.columns:
            year_range = st.slider(
                "Filter by Year Range:",
                min_value=int(working_df['year'].min()),
                max_value=int(working_df['year'].max()),
                value=(int(working_df['year'].min()), int(working_df['year'].max()))
            )
        else:
            year_range = None
    
    # Apply filters
    filtered_df = working_df.copy()
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    
    if year_range and 'year' in working_df.columns:
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) & 
            (filtered_df['year'] <= year_range[1])
        ]
    
    st.markdown(f"**Filtered Data Size:** {len(filtered_df):,} records")
    
    # Preview
    st.markdown("### 👁️ Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)
    
    # Download button
    st.markdown("### 💾 Download")
    
    csv = filtered_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"{dataset_choice.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()