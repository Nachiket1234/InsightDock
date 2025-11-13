# duckdb_setup_fixed.py
import duckdb
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OlistDuckDBSetup:
    def __init__(self, data_dir="/content", db_path="olist.duckdb"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to DuckDB (in-memory + persistent)"""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            return False
    
    def load_csv_files(self):
        """Load all 9 Olist CSV files into DuckDB TABLES (not views)"""
        csv_files = {
            'customers': 'olist_customers_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv', 
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }
        
        for table_name, csv_file in csv_files.items():
            csv_path = self.data_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue
                
            try:
                # Create TABLE using DuckDB's native CSV reader
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{csv_path}')
                """)
                logger.info(f"Created table: {table_name} from {csv_file}")
                
                # Get row count for verification
                count_result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                logger.info(f"  -> Loaded {count_result[0]:,} rows")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
    
    def create_master_view(self):
        """Create a comprehensive master view with proper joins to avoid Cartesian product"""
        try:
            # First, let's create a cleaner master view without the many-to-many relationships
            self.conn.execute("""
                CREATE OR REPLACE VIEW master_olist_clean AS
                SELECT 
                    -- Order details
                    o.order_id, o.order_status, 
                    o.order_purchase_timestamp, o.order_approved_at,
                    o.order_delivered_carrier_date, o.order_delivered_customer_date,
                    o.order_estimated_delivery_date,
                    
                    -- Customer details  
                    c.customer_id, c.customer_unique_id,
                    c.customer_zip_code_prefix, c.customer_city, c.customer_state,
                    
                    -- Product details
                    p.product_id, p.product_category_name,
                    COALESCE(ct.product_category_name_english, p.product_category_name) as product_category_name_english,
                    p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
                    
                    -- Order item details
                    oi.order_item_id, oi.seller_id, oi.shipping_limit_date,
                    oi.price, oi.freight_value,
                    (oi.price + oi.freight_value) as total_amount,
                    
                    -- Seller details
                    s.seller_zip_code_prefix, s.seller_city, s.seller_state
                    
                FROM orders o
                LEFT JOIN customers c ON o.customer_id = c.customer_id
                LEFT JOIN order_items oi ON o.order_id = oi.order_id
                LEFT JOIN products p ON oi.product_id = p.product_id
                LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
                LEFT JOIN sellers s ON oi.seller_id = s.seller_id
            """)
            logger.info("Created master_olist_clean view")
            
            # Verify master view
            count_result = self.conn.execute("SELECT COUNT(*) FROM master_olist_clean").fetchone()
            logger.info(f"Clean master view contains {count_result[0]:,} rows")
            
        except Exception as e:
            logger.error(f"Error creating master view: {e}")
    
    def create_indexes(self):
        """Create performance indexes on key columns - now on TABLES"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_orders_id ON orders(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_purchase_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_customers_id ON customers(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_customers_unique ON customers(customer_unique_id)",
            "CREATE INDEX IF NOT EXISTS idx_products_id ON products(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products(product_category_name)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_geolocation_zip ON geolocation(geolocation_zip_code_prefix)"
        ]
        
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
                logger.info(f"Created index: {index_sql.split(' ON ')[1]}")
            except Exception as e:
                logger.error(f"Error creating index: {e}")
    
    def create_analytical_views(self):
        """Create additional analytical views for common queries"""
        try:
            # Monthly revenue by category
            self.conn.execute("""
                CREATE OR REPLACE VIEW monthly_category_revenue AS
                SELECT 
                    DATE_TRUNC('month', order_purchase_timestamp) as month,
                    product_category_name_english as category,
                    COUNT(DISTINCT order_id) as order_count,
                    COUNT(*) as item_count,
                    ROUND(SUM(total_amount), 2) as total_revenue,
                    ROUND(AVG(total_amount), 2) as avg_order_value
                FROM master_olist_clean
                WHERE order_purchase_timestamp IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 5 DESC
            """)
            logger.info("Created monthly_category_revenue view")
            
            # Customer analytics view
            self.conn.execute("""
                CREATE OR REPLACE VIEW customer_analytics AS
                SELECT 
                    customer_unique_id,
                    COUNT(DISTINCT order_id) as total_orders,
                    ROUND(SUM(total_amount), 2) as total_spent,
                    ROUND(AVG(total_amount), 2) as avg_order_value,
                    MIN(order_purchase_timestamp) as first_order_date,
                    MAX(order_purchase_timestamp) as last_order_date
                FROM master_olist_clean
                GROUP BY customer_unique_id
                HAVING COUNT(DISTINCT order_id) >= 1
            """)
            logger.info("Created customer_analytics view")
            
            # Geographic sales view
            self.conn.execute("""
                CREATE OR REPLACE VIEW geographic_sales AS
                SELECT 
                    customer_state,
                    customer_city,
                    COUNT(DISTINCT order_id) as order_count,
                    ROUND(SUM(total_amount), 2) as total_revenue,
                    COUNT(DISTINCT customer_unique_id) as unique_customers
                FROM master_olist_clean
                GROUP BY customer_state, customer_city
                ORDER BY total_revenue DESC
            """)
            logger.info("Created geographic_sales view")
            
        except Exception as e:
            logger.error(f"Error creating analytical views: {e}")
    
    def sync_to_motherduck(self):
        """Sync database to MotherDuck"""
        md_token = os.getenv('MD_TOKEN') or os.getenv('DUCKDB_MD_TOKEN')
        if not md_token:
            logger.warning("MD_TOKEN not found. Skipping MotherDuck sync.")
            return False
            
        try:
            # Install and load motherduck extension
            self.conn.execute("INSTALL motherduck")
            self.conn.execute("LOAD motherduck")
            
            # Connect to MotherDuck and attach
            self.conn.execute(f"ATTACH 'md:olist_brazilian_ecommerce?token={md_token}' AS motherduck_db")
            logger.info("Connected to MotherDuck")
            
            # Sync all tables to MotherDuck
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()
            
            for (table_name,) in tables:
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE motherduck_db.main.{table_name} AS 
                    SELECT * FROM main.{table_name}
                """)
                logger.info(f"Synced table {table_name} to MotherDuck")
            
            # Sync views to MotherDuck
            views = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' AND table_type = 'VIEW'
            """).fetchall()
            
            for (view_name,) in views:
                # Get view definition and recreate in MotherDuck
                view_def = self.conn.execute(f"""
                    SELECT sql FROM duckdb_views() WHERE view_name = '{view_name}'
                """).fetchone()
                
                if view_def:
                    self.conn.execute(f"CREATE OR REPLACE VIEW motherduck_db.main.{view_name} AS {view_def[0]}")
                    logger.info(f"Synced view {view_name} to MotherDuck")
            
            self.conn.execute("DETACH motherduck_db")
            logger.info("Completed MotherDuck sync")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing to MotherDuck: {e}")
            return False
    
    def run_test_queries(self):
        """Run test queries to verify setup"""
        test_queries = [
            "SELECT COUNT(*) as total_orders FROM orders",
            "SELECT COUNT(DISTINCT customer_unique_id) as unique_customers FROM customers",
            "SELECT COUNT(DISTINCT product_id) as unique_products FROM products",
            "SELECT COUNT(*) as clean_master_rows FROM master_olist_clean",
            """
            SELECT 
                product_category_name_english as category,
                COUNT(*) as order_count,
                ROUND(SUM(total_amount), 2) as total_revenue
            FROM master_olist_clean 
            WHERE product_category_name_english IS NOT NULL
            GROUP BY product_category_name_english 
            ORDER BY total_revenue DESC 
            LIMIT 5
            """
        ]
        
        logger.info("Running test queries...")
        for i, query in enumerate(test_queries, 1):
            try:
                result = self.conn.execute(query).fetchall()
                logger.info(f"Test {i}: {result[0] if result else 'No results'}")
            except Exception as e:
                logger.error(f"Test query {i} failed: {e}")
    
    def setup_complete(self):
        """Main setup method that runs all steps"""
        logger.info("Starting Olist DuckDB setup...")
        
        if not self.connect():
            return False
            
        self.load_csv_files()
        self.create_indexes()  # Create indexes BEFORE views for better performance
        self.create_master_view() 
        self.create_analytical_views()
        self.sync_to_motherduck()
        self.run_test_queries()
        
        logger.info("Olist DuckDB setup completed successfully!")
        return True

# Fix the database summary query in the main() function
def main():
    """Main executable function"""
    setup = OlistDuckDBSetup(
        data_dir="/content",
        db_path="olist_fixed.duckdb"
    )
    
    if setup.setup_complete():
        print("✅ Setup completed successfully!")
        print(f"📊 Database saved to: {setup.db_path}")
        
        # Show final database info - FIXED VERSION
        if setup.conn:
            # Get table names
            tables = setup.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()
            
            print("\n📋 Database Tables Summary:")
            for (table_name,) in tables:
                count_result = setup.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                print(f"  {table_name}: {count_result[0]:,} rows")
                
            # Count views
            views_count = setup.conn.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'main' AND table_type = 'VIEW'
            """).fetchone()[0]
            print(f"\n  Analytical views: {views_count}")
            
    else:
        print("❌ Setup failed. Check logs for details.")
        return 1
    
    # Close connection
    if setup.conn:
        setup.conn.close()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())