import pe_analysis

def main():
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    input_data, business_applications_target_data = pe_analysis.process_data(input_data_file_names, "total_construction_spending_health_care.csv", 2005, 2025)
    print("TARGET DATA: CONSTRUCTION SPENDING")
    pe_analysis.run_evaluation(input_data, business_applications_target_data)
    print("TARGET DATA: CONSTRUCTION SPENDING - END")

main()
