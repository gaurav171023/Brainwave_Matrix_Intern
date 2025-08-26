import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("Starting sales data generation...")

# Define product categories and products
categories = {
    'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch', 'Camera'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Dress', 'Hoodie'],
    'Home & Garden': ['Furniture', 'Bedding', 'Kitchen Appliances', 'Decor', 'Plants', 'Tools'],
    'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Children Books', 'Comics', 'Biography'],
    'Sports': ['Running Shoes', 'Gym Equipment', 'Sportswear', 'Bicycle', 'Football', 'Tennis Racket']
}

# Product prices (base prices)
product_prices = {
    'Laptop': 800, 'Smartphone': 600, 'Tablet': 300, 'Headphones': 100, 'Smartwatch': 250, 'Camera': 400,
    'T-Shirt': 25, 'Jeans': 60, 'Jacket': 120, 'Sneakers': 80, 'Dress': 45, 'Hoodie': 40,
    'Furniture': 300, 'Bedding': 80, 'Kitchen Appliances': 150, 'Decor': 50, 'Plants': 20, 'Tools': 35,
    'Fiction': 15, 'Non-Fiction': 20, 'Textbook': 100, 'Children Books': 12, 'Comics': 8, 'Biography': 18,
    'Running Shoes': 90, 'Gym Equipment': 200, 'Sportswear': 35, 'Bicycle': 400, 'Football': 25, 'Tennis Racket': 60
}

# Regions and their sales multipliers
regions = {
    'North': 1.2, 'South': 1.0, 'East': 1.1, 'West': 0.9, 'Central': 1.05
}

# Sales channels
channels = ['Online', 'In-Store', 'Mobile App']

# Customer segments
customer_segments = ['Premium', 'Regular', 'Budget']

def generate_sales_data(num_records=5000):
    """Generate synthetic sales data"""
    
    print(f"Generating {num_records} sales records...")
    
    # Date range: last 2 years
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    
    data = []
    
    for i in range(num_records):
        if i % 1000 == 0:
            print(f"Generated {i} records...")
            
        # Generate random date
        random_date = start_date + timedelta(days=random.randint(0, 730))
        
        # Choose category and product
        category = random.choice(list(categories.keys()))
        product = random.choice(categories[category])
        
        # Base price with some variation
        base_price = product_prices[product]
        price_variation = random.uniform(0.8, 1.3)  # ±30% variation
        unit_price = round(base_price * price_variation, 2)
        
        # Quantity (influenced by price - cheaper items sold in higher quantities)
        if base_price < 50:
            quantity = random.randint(1, 8)
        elif base_price < 200:
            quantity = random.randint(1, 4)
        else:
            quantity = random.randint(1, 2)
        
        # Region
        region = random.choice(list(regions.keys()))
        
        # Apply seasonal effects
        month = random_date.month
        seasonal_multiplier = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.4
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.2
        elif month in [1, 2]:  # Post-holiday
            seasonal_multiplier = 0.8
        
        # Apply region multiplier
        region_multiplier = regions[region]
        
        # Final price calculation
        final_unit_price = round(unit_price * seasonal_multiplier * region_multiplier, 2)
        total_sales = round(final_unit_price * quantity, 2)
        
        # Generate other fields
        customer_id = f"CUST_{random.randint(1000, 9999)}"
        order_id = f"ORD_{random.randint(100000, 999999)}"
        sales_channel = random.choice(channels)
        customer_segment = random.choice(customer_segments)
        
        # Cost (60-80% of selling price for profit calculation)
        cost_ratio = random.uniform(0.6, 0.8)
        total_cost = round(total_sales * cost_ratio, 2)
        profit = round(total_sales - total_cost, 2)
        
        # Create record
        record = {
            'Order_ID': order_id,
            'Date': random_date.strftime('%Y-%m-%d'),
            'Customer_ID': customer_id,
            'Product': product,
            'Category': category,
            'Quantity': quantity,
            'Unit_Price': final_unit_price,
            'Total_Sales': total_sales,
            'Total_Cost': total_cost,
            'Profit': profit,
            'Region': region,
            'Sales_Channel': sales_channel,
            'Customer_Segment': customer_segment
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

# Generate the dataset
print("=" * 50)
print("SALES DATA GENERATOR - BRAINWAVE MATRIX SOLUTIONS")
print("=" * 50)

sales_df = generate_sales_data(5000)

# Display basic info
print(f"\n✅ Dataset created successfully!")
print(f"📊 Total records: {len(sales_df):,}")
print(f"📅 Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
print(f"💰 Total sales value: ${sales_df['Total_Sales'].sum():,.2f}")
print(f"📈 Total profit: ${sales_df['Profit'].sum():,.2f}")

# Save to CSV
sales_df.to_csv('data/sales_data.csv', index=False)
print(f"💾 Data saved to 'data/sales_data.csv'")

print("\n" + "=" * 50)
print("SAMPLE DATA PREVIEW")
print("=" * 50)
print(sales_df.head(10))

print("\n" + "=" * 50)
print("DATASET SUMMARY STATISTICS")
print("=" * 50)
print(sales_df.describe().round(2))

print("\n✅ Data generation completed successfully!")
print("🎯 Ready for analysis!")