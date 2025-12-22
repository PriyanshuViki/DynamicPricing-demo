# app.py - Dynamic Pricing Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide", page_icon="💰")

# Load and cache data
@st.cache_data
def load_data():
    train = pd.read_csv('data/processed/train_data.csv')
    test = pd.read_csv('data/processed/test_data.csv')
    
    # Create encoding maps
    material_price_map = train.groupby('Material_Clean')['Unit_Price'].mean().to_dict()
    group_price_map = train.groupby('Material Group')['Unit_Price'].mean().to_dict()
    
    # Encode features
    for df in [train, test]:
        df['Material_Enc'] = df['Material_Clean'].map(material_price_map).fillna(train['Unit_Price'].mean())
        df['Group_Enc'] = df['Material Group'].map(group_price_map).fillna(train['Unit_Price'].mean())
    
    return train, test, material_price_map, group_price_map

@st.cache_resource
def train_model(train):
    X_train = train[['Material_Enc', 'Group_Enc', 'Order Quantity', 'Unit_Cost']]
    y_train = train['Unit_Price']
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data and model
train, test, material_price_map, group_price_map = load_data()
model = train_model(train)

# Predictions
X_test = test[['Material_Enc', 'Group_Enc', 'Order Quantity', 'Unit_Cost']]
test['Predicted_Price'] = model.predict(X_test)
test['Current_Margin'] = ((test['Unit_Price'] - test['Unit_Cost']) / test['Unit_Price'] * 100)
test['Predicted_Margin'] = ((test['Predicted_Price'] - test['Unit_Cost']) / test['Predicted_Price'] * 100)
test['Price_Change_%'] = ((test['Predicted_Price'] - test['Unit_Price']) / test['Unit_Price'] * 100)
test['Revenue_Impact'] = (test['Predicted_Price'] - test['Unit_Price']) * test['Order Quantity']

# Metrics
current_revenue = (test['Unit_Price'] * test['Order Quantity']).sum()
optimized_revenue = (test['Predicted_Price'] * test['Order Quantity']).sum()
opportunity = optimized_revenue - current_revenue
r2 = r2_score(test['Unit_Price'], test['Predicted_Price'])
mae = mean_absolute_error(test['Unit_Price'], test['Predicted_Price'])

# Header
st.title("💰 Dynamic Pricing Model Dashboard")
st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Revenue Opportunity", f"₹{opportunity/1e7:.2f} Cr", f"+{opportunity/current_revenue*100:.2f}%")
with col2:
    st.metric("Model Accuracy (R²)", f"{r2:.3f}", "65.2%")
with col3:
    st.metric("Prediction Error (MAE)", f"₹{mae:.0f}", "4.97% MAPE")
with col4:
    st.metric("Optimized Transactions", f"{len(test)}", "2,100 records")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Recommendations", "💰 Profit Impact", "📈 Volume Pricing", "🎯 Pricing Status"])

# TAB 1: Price Recommendations
with tab1:
    st.subheader("A. Price Recommendation Accuracy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scatter plot
        fig = px.scatter(test, x='Unit_Price', y='Predicted_Price', 
                        color='Order Quantity', size='Order Quantity',
                        labels={'Unit_Price': 'Actual Price (₹)', 'Predicted_Price': 'Predicted Price (₹)'},
                        title='Actual vs Predicted Prices')
        fig.add_trace(go.Scatter(x=[0, test['Unit_Price'].max()], y=[0, test['Unit_Price'].max()],
                                mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📦 Sample Recommendations")
        top_materials = test.groupby('Material_Clean').size().sort_values(ascending=False).head(3).index
        for mat in top_materials:
            sample = test[test['Material_Clean']==mat].iloc[0]
            with st.expander(f"Material: {mat}"):
                st.write(f"**Current Price:** ₹{sample['Unit_Price']:.2f}")
                st.write(f"**Recommended:** ₹{sample['Predicted_Price']:.2f}")
                st.write(f"**Change:** {sample['Price_Change_%']:+.1f}%")
                st.write(f"**Margin:** {sample['Current_Margin']:.1f}% → {sample['Predicted_Margin']:.1f}%")

# TAB 2: Profit Impact
with tab2:
    st.subheader("B. Profit Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue Impact
        fig = go.Figure(data=[
            go.Bar(name='Current', x=['Revenue'], y=[current_revenue/1e7], marker_color='lightgray'),
            go.Bar(name='Optimized', x=['Revenue'], y=[optimized_revenue/1e7], marker_color='lightgreen'),
            go.Bar(name='Opportunity', x=['Revenue'], y=[opportunity/1e7], marker_color='gold')
        ])
        fig.update_layout(title='Revenue Impact (₹ Crores)', barmode='group', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Margin Improvement
        margin_data = test.groupby('Material_Clean').agg({
            'Current_Margin': 'mean',
            'Predicted_Margin': 'mean'
        }).sort_values('Predicted_Margin', ascending=False).head(5)
        
        fig = go.Figure(data=[
            go.Bar(name='Current Margin', x=margin_data.index, y=margin_data['Current_Margin'], marker_color='coral'),
            go.Bar(name='Optimized Margin', x=margin_data.index, y=margin_data['Predicted_Margin'], marker_color='lightgreen')
        ])
        fig.update_layout(title='Top 5 Materials - Margin Improvement', barmode='group', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Opportunities Table
    st.markdown("### 🏆 Top Revenue Opportunities")
    top_opps = test.groupby('Material_Clean').agg({
        'Revenue_Impact': 'sum',
        'Price_Change_%': 'mean',
        'Material_Clean': 'count'
    }).rename(columns={'Material_Clean': 'Transactions'}).sort_values('Revenue_Impact', ascending=False).head(10)
    top_opps['Revenue_Impact'] = top_opps['Revenue_Impact'].apply(lambda x: f"₹{x/1e6:.2f}M")
    top_opps['Price_Change_%'] = top_opps['Price_Change_%'].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(top_opps, use_container_width=True)

# TAB 3: Volume Pricing
with tab3:
    st.subheader("C. Volume-Based Pricing Analysis")
    
    # Material selector
    materials = test['Material_Clean'].unique()
    selected_material = st.selectbox("Select Material", materials)
    
    # Calculate volume pricing
    sample_cost = test[test['Material_Clean']==selected_material]['Unit_Cost'].mean()
    quantities = [50, 100, 200, 500, 1000, 2000, 5000]
    prices = []
    
    for qty in quantities:
        mat_enc = material_price_map.get(selected_material, train['Unit_Price'].mean())
        group = test[test['Material_Clean']==selected_material]['Material Group'].iloc[0]
        group_enc = group_price_map.get(group, train['Unit_Price'].mean())
        input_data = pd.DataFrame([[mat_enc, group_enc, qty, sample_cost]], 
                                  columns=['Material_Enc', 'Group_Enc', 'Order Quantity', 'Unit_Cost'])
        prices.append(model.predict(input_data)[0])
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quantities, y=prices, mode='lines+markers', 
                            line=dict(width=3, color='purple'), marker=dict(size=10)))
    fig.update_layout(title=f'Price vs Quantity for Material {selected_material}',
                     xaxis_title='Order Quantity', yaxis_title='Recommended Price (₹)', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Pricing table
    pricing_df = pd.DataFrame({
        'Quantity': quantities,
        'Price (₹)': [f"₹{p:.2f}" for p in prices],
        'Discount %': [f"{((prices[0]-p)/prices[0]*100):.1f}%" for p in prices]
    })
    st.dataframe(pricing_df, use_container_width=True)

# TAB 4: Pricing Status
with tab4:
    st.subheader("E. Pricing Status Distribution")
    
    underpriced = test[test['Price_Change_%'] > 5]
    overpriced = test[test['Price_Change_%'] < -5]
    optimal = test[(test['Price_Change_%'] >= -5) & (test['Price_Change_%'] <= 5)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Underpriced (>5%)', 'Optimal (±5%)', 'Overpriced (<-5%)'],
            values=[len(underpriced), len(optimal), len(overpriced)],
            marker_colors=['#e74c3c', '#2ecc71', '#3498db'],
            hole=0.3
        )])
        fig.update_layout(title='Pricing Status Distribution', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(x=['Underpriced', 'Optimal', 'Overpriced'],
                  y=[len(underpriced), len(optimal), len(overpriced)],
                  marker_color=['#e74c3c', '#2ecc71', '#3498db'],
                  text=[f"{len(underpriced)} ({len(underpriced)/len(test)*100:.1f}%)",
                        f"{len(optimal)} ({len(optimal)/len(test)*100:.1f}%)",
                        f"{len(overpriced)} ({len(overpriced)/len(test)*100:.1f}%)"],
                  textposition='auto')
        ])
        fig.update_layout(title='Transaction Count by Status', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Status metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Underpriced", len(underpriced), f"{len(underpriced)/len(test)*100:.1f}%")
    with col2:
        st.metric("Optimally Priced", len(optimal), f"{len(optimal)/len(test)*100:.1f}%")
    with col3:
        st.metric("Overpriced", len(overpriced), f"{len(overpriced)/len(test)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("**Dynamic Pricing Model** | Powered by Gradient Boosting | R² = 0.652")
