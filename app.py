import streamlit as st
import sqlite3
import pandas as pd
import random
import hashlib
import base64
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

# --- CONFIG ---
DB_PATH = "agentiq.db"
LOGO_PATH = "logo.png"
SESSION_STATE_KEY = "recommendation_history"

# --- DATABASE MANAGEMENT ---
@contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper closure."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        yield conn
    finally:
        if conn:
            conn.close()

def execute_query(query: str, params: tuple = (), fetch_all: bool = True) -> Any:
    """Execute a database query with proper error handling."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch_all:
                return cursor.fetchall()
            return cursor.fetchone()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return [] if fetch_all else None

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=300)
def load_customer_ids() -> List[str]:
    """Load all customer IDs from the database with caching."""
    result = execute_query("SELECT Customer_ID FROM customers")
    return [row['Customer_ID'] for row in result] if result else []

def get_customer_details(customer_id: str) -> Optional[Dict]:
    """Get customer details by ID."""
    result = execute_query(
        "SELECT * FROM customers WHERE Customer_ID = ?", 
        (customer_id,), 
        fetch_all=False
    )
    return dict(result) if result else None

@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame:
    """Load all products with caching and proper type conversion."""
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT * FROM products", conn)
    
    # Convert price to proper numeric type
    try:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        # Fill NaN values with 0 if any conversion failed
        df['Price'] = df['Price'].fillna(0)
    except Exception as e:
        st.warning(f"Warning: Could not convert all prices to numeric values. Some features may be affected. Error: {e}")
    
    return df

@st.cache_data(ttl=600)
def get_unique_categories() -> List[str]:
    """Get unique product categories with caching."""
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT DISTINCT Category FROM products", conn)
    return ['All'] + sorted(df['Category'].dropna().unique().tolist())

def parse_browsing_history(history_str: str) -> List[str]:
    """Safely parse browsing history string."""
    if not history_str:
        return []
    
    try:
        # Use json.loads instead of eval for safety
        if history_str.startswith('[') and history_str.endswith(']'):
            return json.loads(history_str)
        else:
            return [history_str]
    except json.JSONDecodeError:
        # If not valid JSON, treat as a single category
        return [history_str] if history_str else []

def match_browsing_categories(customer: Dict, products: pd.DataFrame) -> pd.DataFrame:
    """Match products to customer browsing history."""
    browsing = parse_browsing_history(customer.get('Browsing_History', ''))
    
    if not browsing:
        return products
    
    matched = products[(products['Category'].isin(browsing)) | 
                      (products['Subcategory'].isin(browsing))]
    
    return matched if not matched.empty else products

def get_diverse_recommendations(customer_id: str, items_df: pd.DataFrame, customer_data: Dict, k: int = 4) -> pd.DataFrame:
    """Get highly accurate and personalized product recommendations.
    
    Uses a sophisticated multi-factor scoring algorithm that considers:
    1. Product category affinity based on browsing history
    2. Price point alignment with customer segment
    3. Demographic relevance based on age, location
    4. Product rating and popularity
    5. Previous recommendation history to avoid repetition
    """
    if items_df.empty:
        return pd.DataFrame()
    
    # Initialize session state for tracking recommendation history
    if SESSION_STATE_KEY not in st.session_state:
        st.session_state[SESSION_STATE_KEY] = {}
    
    if customer_id not in st.session_state[SESSION_STATE_KEY]:
        st.session_state[SESSION_STATE_KEY][customer_id] = []
    
    # Get previously recommended product IDs for this customer
    previous_recommendations = st.session_state[SESSION_STATE_KEY][customer_id]
    
    # Create a copy of the dataframe to avoid modifying the original
    scored_products = items_df.copy()
    
    # Calculate affinity score for each product
    # Higher score = better match for the customer
    
    # FACTOR 1: Category affinity based on browsing history (0-5 points)
    browsing_history = parse_browsing_history(customer_data.get('Browsing_History', ''))
    
    def calculate_category_score(row):
        # Direct match with browsing history gets highest score
        if row['Category'] in browsing_history:
            return 5
        # Subcategory match gets second highest
        elif row['Subcategory'] in browsing_history:
            return 4
        # Look for partial matches in browsing history categories
        for category in browsing_history:
            # Check if the category name is contained in another (e.g., "Kitchen" in "Kitchen Appliances")
            if (isinstance(category, str) and isinstance(row['Category'], str) and 
                (category.lower() in row['Category'].lower() or row['Category'].lower() in category.lower())):
                return 3
        return 0
    
    scored_products['category_score'] = scored_products.apply(calculate_category_score, axis=1)
    
    # FACTOR 2: Price point alignment with customer segment (0-3 points)
    customer_segment = customer_data.get('Customer_Segment', '')
    
    def calculate_price_score(row):
        try:
            price = float(row['Price'])
            if customer_segment == 'Premium':
                # Premium customers prefer higher-priced items
                if price > 5000:
                    return 3
                elif price > 2000:
                    return 2
                elif price > 1000:
                    return 1
            elif customer_segment == 'Budget':
                # Budget customers prefer lower-priced items
                if price < 500:
                    return 3
                elif price < 1000:
                    return 2
                elif price < 2000:
                    return 1
            else:
                # Mid-range customers prefer mid-range prices
                if 1000 <= price <= 3000:
                    return 3
                elif 500 <= price <= 4000:
                    return 2
                else:
                    return 1
        except (ValueError, TypeError):
            return 0
        return 0
    
    scored_products['price_score'] = scored_products.apply(calculate_price_score, axis=1)
    
    # FACTOR 3: Demographic relevance (0-3 points)
    age = customer_data.get('Age', 0)
    location = customer_data.get('Location', '')
    
    def calculate_demographic_score(row):
        score = 0
        
        # Age-based preferences
        if age < 25 and row['Category'] in ['Electronics', 'Fashion', 'Gaming']:
            score += 1
        elif 25 <= age < 40 and row['Category'] in ['Home & Kitchen', 'Electronics', 'Fitness']:
            score += 1
        elif age >= 40 and row['Category'] in ['Health', 'Home & Kitchen', 'Books']:
            score += 1
            
        # Location-based preferences
        if location == 'Urban' and row['Category'] in ['Electronics', 'Fashion', 'Dining']:
            score += 1
        elif location == 'Suburban' and row['Category'] in ['Home & Kitchen', 'Garden', 'Appliances']:
            score += 1
        elif location == 'Rural' and row['Category'] in ['Tools', 'Outdoors', 'Garden']:
            score += 1
            
        # Additional affinity for specific subcategories
        target_subcategories = []
        
        if age < 25:
            target_subcategories = ['Smartphones', 'Headphones', 'Sneakers']
        elif 25 <= age < 40:
            target_subcategories = ['Coffee Makers', 'Laptops', 'Fitness Trackers']
        else:
            target_subcategories = ['Health Monitors', 'Kitchen Appliances', 'Books']
            
        if isinstance(row['Subcategory'], str) and row['Subcategory'] in target_subcategories:
            score += 1
            
        return min(score, 3)  # Cap at 3 points
    
    scored_products['demographic_score'] = scored_products.apply(calculate_demographic_score, axis=1)
    
    # FACTOR 4: Product rating and quality (0-2 points)
    def calculate_rating_score(row):
        try:
            rating = float(row['Product_Rating'])
            if rating >= 4.5:
                return 2
            elif rating >= 4.0:
                return 1
            else:
                return 0
        except (ValueError, TypeError):
            return 0
    
    scored_products['rating_score'] = scored_products.apply(calculate_rating_score, axis=1)
    
    # FACTOR 5: Novelty - prioritize items not previously recommended (-3 to 0 points)
    def calculate_novelty_score(row):
        if row['Product_ID'] in previous_recommendations:
            return -3  # Strong penalty for previously shown products
        return 0
    
    scored_products['novelty_score'] = scored_products.apply(calculate_novelty_score, axis=1)
    
    # Calculate total recommendation score (max 13 points)
    scored_products['total_score'] = (
        scored_products['category_score'] + 
        scored_products['price_score'] +
        scored_products['demographic_score'] + 
        scored_products['rating_score'] +
        scored_products['novelty_score']
    )
    
    # Sort by score (highest first) and get top k recommendations
    recommended = scored_products.sort_values('total_score', ascending=False).head(k)
    
    # If we got fewer than k recommendations (unlikely but possible), 
    # fill with random products not previously recommended
    if len(recommended) < k:
        remaining_products = items_df[~items_df['Product_ID'].isin(recommended['Product_ID'])]
        if not remaining_products.empty:
            additional = remaining_products.sample(min(k - len(recommended), len(remaining_products)))
            recommended = pd.concat([recommended, additional])
    
    # Ensure diversity if we have enough products
    if len(recommended) > 2 and len(items_df) > k*2:
        category_counts = recommended['Category'].value_counts()
        most_common_category = category_counts.index[0]
        
        # If more than half of recommendations are from the same category, replace one
        if category_counts.iloc[0] > k/2:
            # Find recommendation(s) to replace
            to_replace = recommended[recommended['Category'] == most_common_category].index[0]
            
            # Find a replacement from a different category
            possible_replacements = items_df[
                ~items_df['Product_ID'].isin(recommended['Product_ID']) & 
                (items_df['Category'] != most_common_category)
            ]
            
            if not possible_replacements.empty:
                replacement = possible_replacements.sample(1)
                recommended = recommended.drop(to_replace).append(replacement)
    
    # Update recommendation history
    new_recommendations = recommended['Product_ID'].tolist()
    st.session_state[SESSION_STATE_KEY][customer_id] = (previous_recommendations + new_recommendations)[-20:]  # Keep last 20
    
    # Save scores for explanation
    st.session_state['last_recommendation_scores'] = recommended[['Product_ID', 'total_score', 'category_score', 
                                                               'price_score', 'demographic_score', 'rating_score']]
    
    # Return recommendations without the scoring columns
    return recommended.drop(columns=['category_score', 'price_score', 'demographic_score', 
                                      'rating_score', 'novelty_score', 'total_score'])

def generate_personalized_reason(product: Dict, customer: Dict) -> str:
    """Generate highly personalized recommendation reason based on multiple factors."""
    # Check if we have scoring data from the recommendation algorithm
    if 'last_recommendation_scores' in st.session_state:
        scores_df = st.session_state['last_recommendation_scores']
        product_scores = scores_df[scores_df['Product_ID'] == product['Product_ID']]
        
        if not product_scores.empty:
            # Get individual factor scores
            category_score = product_scores['category_score'].iloc[0]
            price_score = product_scores['price_score'].iloc[0]
            demographic_score = product_scores['demographic_score'].iloc[0]
            rating_score = product_scores['rating_score'].iloc[0]
            
            # Determine the strongest factor for personalization
            max_score = max(category_score, price_score, demographic_score, rating_score)
            
            # Generate reason based on strongest factor
            if max_score == category_score and category_score > 0:
                browsing = parse_browsing_history(customer.get('Browsing_History', ''))
                if product['Category'] in browsing:
                    return f"🧠 Perfect match for your interest in {product['Category']}!"
                elif product['Subcategory'] in browsing:
                    return f"🧠 Exactly what you've been browsing in {product['Subcategory']}!"
                else:
                    return f"🧠 Aligns perfectly with your browsing history and preferences!"
                    
            elif max_score == price_score and price_score > 0:
                if customer.get('Customer_Segment') == 'Premium':
                    return f"💎 Premium quality product that matches your excellent taste!"
                elif customer.get('Customer_Segment') == 'Budget':
                    return f"💰 Great value that fits your smart shopping preferences!"
                else:
                    return f"⚖️ Perfect price point based on your purchasing patterns!"
                    
            elif max_score == demographic_score and demographic_score > 0:
                age = customer.get('Age', 0)
                location = customer.get('Location', '')
                
                if age < 25:
                    return f"🚀 Popular choice among trendsetters in your age group in {location}!"
                elif 25 <= age < 40:
                    return f"👌 Top pick for {location} shoppers in their {age}s like you!"
                else:
                    return f"🎯 Specially curated for discerning {location} shoppers in your demographic!"
                    
            elif max_score == rating_score and rating_score > 0:
                return f"⭐ Highly rated ({product['Product_Rating']}/5) product our AI thinks you'll love!"
    
    # Fallback to simpler personalization if no scoring data available
    browsing = parse_browsing_history(customer.get('Browsing_History', ''))
    
    # Check if product matches browsing history
    if product['Category'] in browsing or product['Subcategory'] in browsing:
        return f"🧠 You're into {product['Category']} — and this is a perfect match!"
    
    # Generate reason based on customer attributes
    age = customer.get('Age', 0)
    location = customer.get('Location', '')
    segment = customer.get('Customer_Segment', '')
    
    reasons = [
        f"💖 Selected for {segment} shoppers in {location} like you!",
        f"🎯 Matches the preferences of {age}-year-old shoppers!",
        f"🌟 AI-picked especially for your unique shopping profile!",
        f"✨ Our algorithm detected your interest in {product['Category']}!"
    ]
    
    return random.choice(reasons)

# Diverse set of fun facts to avoid repetition
fun_facts = [
    "🎉 Top-rated by users who also loved your past picks!",
    "🚀 Flying off the shelves — blink and it's gone!",
    "💡 Gen Z-approved and AI-recommended!",
    "🌟 A 5-star pick for a 5-star shopper like you!",
    "✨ Featured in our monthly top-selling products!",
    "💯 9 out of 10 similar customers loved this item!",
    "🏆 Award-winning quality at a price you'll love!",
    "⚡ This item has been trending for the past week!",
    "🌈 A perfect addition to complete your collection!",
    "🧠 Smart shoppers are adding this to their carts!",
    "🎁 Makes an excellent gift for someone special!",
    "💪 Built to last with premium materials!",
    "🔥 One of our hottest items this season!",
    "🌱 Eco-friendly choice for sustainable shopping!",
    "💎 Hidden gem discovered by our recommendation algorithm!"
]

# --- UI COMPONENTS ---
def get_base64_image(image_path: str) -> str:
    """Convert an image to base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found at {image_path}. Using text header instead.")
        return ""

def render_header():
    """Render the app header with logo."""
    # Apply gradient background and styling
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #d299c2, #fef9d7);
            }    
            .css-18e3th9 {
                background-color: rgba(255, 255, 255, 0.05);
                padding: 2rem;
                border-radius: 1rem;
            }
            h1 {
                text-align: center;
                color: #5f0a87;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }
            .recommendation-card {
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .recommendation-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            }
            .price-tag {
                font-size: 1.4rem;
                font-weight: bold;
                color: #5f0a87;
            }
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
                color: white;
                background: linear-gradient(45deg, #a4508b, #5f0a87);
                margin-right: 5px;
            }
            .customer-info {
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                padding: 10px 15px;
                margin-bottom: 20px;
            }
            .filters-container {
                background-color: rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Try to load and display logo
    logo_base64 = get_base64_image(LOGO_PATH)
    if logo_base64:
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{logo_base64}' 
                    style='width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'/>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1>AgentIQ — Your Cart's Got PhD Now</h1>", unsafe_allow_html=True)

def render_product_card(product, customer):
    """Render a beautiful product recommendation card."""
    # Handle price formatting with proper error checking
    try:
        price = float(product['Price'])
        price_display = f"₹{price:.2f}"
    except (ValueError, TypeError):
        # If conversion fails, display as is
        price_display = f"₹{product['Price']}"
    
    st.markdown(f"""
    <div class="recommendation-card">
        <h3>🛍 {product['Subcategory']}</h3>
        <span class="price-tag">{price_display}</span>
        <div style="margin: 10px 0;">
            <span class="badge">{product['Category']}</span>
            <span>⭐ {product['Product_Rating']}</span>
        </div>
        <p><strong>Brand:</strong> {product['Brand']}</p>
        <p><em>{generate_personalized_reason(product, customer)}</em></p>
        <p>{random.choice(fun_facts)}</p>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    """Main application function."""
    st.set_page_config(
        page_title="AgentIQ - Smart Shopping AI",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    render_header()
    
    # Load customer data
    customer_ids = load_customer_ids()
    
    # Layout for customer selection and filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_id = st.selectbox("👤 Choose your Customer ID", customer_ids)
    
    with col2:
        st.write(" ")  # Spacing
        st.write(" ")  # Spacing
        show_filters = st.checkbox("🔧 Show Filters", value=False)
    
    # Filters section
    if show_filters:
        st.markdown('<div class="filters-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            price_range = st.slider("💰 Price Range (₹)", 50, 10000, (500, 5000), step=50)
            only_browsing = st.checkbox("📚 Match my Browsing History", value=True)
        
        with col2:
            categories = get_unique_categories()
            category_filter = st.selectbox("🔍 Choose a Category", categories)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Default values when filters are hidden
        price_range = (50, 10000)
        category_filter = 'All'
        only_browsing = True
    
    # Button for recommendations
    if st.button("🎯 Get Smart Recommendations", type="primary"):
        with st.spinner("AI is finding perfect matches for you..."):
            customer = get_customer_details(customer_id)
            
            if not customer:
                st.error("Customer not found. Please select a valid customer ID.")
            else:
                # Display customer info
                st.markdown(f"""
                <div class="customer-info">
                    <h3>👤 {customer['Customer_ID']}</h3>
                    <p>Age: {customer['Age']} | Segment: {customer['Customer_Segment']} | Location: {customer['Location']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Filter products
                products = load_products()
                
                if only_browsing:
                    products = match_browsing_categories(customer, products)
                
                if category_filter != 'All':
                    products = products[products['Category'] == category_filter]
                
                # Convert price column to numeric and handle filtering properly
                products['Price'] = pd.to_numeric(products['Price'], errors='coerce')
                products = products.dropna(subset=['Price'])  # Remove rows with non-numeric prices
                products = products[(products['Price'] >= price_range[0]) & 
                                   (products['Price'] <= price_range[1])]
                
                if products.empty:
                    st.warning("No products found matching your filters! Try adjusting your filter criteria.")
                else:
                    # Get diverse recommendations with enhanced accuracy
                    recommended = get_diverse_recommendations(customer_id, products, customer, 4)
                    
                    # Add recommendation explanation
                    if not recommended.empty:
                        st.markdown("### 🌟 Your Smart Recommendations")
                        
                        # Display recommendations in a responsive grid
                        for _, product in recommended.iterrows():
                            render_product_card(product, customer)
                        
                        with st.expander("🔍 How our AI selected these products"):
                            st.markdown("""
                            Our recommendation system analyzed your profile and browsing patterns to find the perfect products for you.
                            We considered:
                            - Your browsing history and category preferences
                            - Price points appropriate for your customer segment
                            - Demographic patterns from similar shoppers
                            - Product ratings and quality metrics
                            - Products you haven't seen before
                            """)
                        
                        # Add a "Try Again" button for more recommendations
                        if st.button("🔄 Get More Recommendations"):
                            st.experimental_rerun()

if __name__ == "__main__":
    main()