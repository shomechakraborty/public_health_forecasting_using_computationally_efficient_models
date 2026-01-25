import pe_analysis_x_single

def main():
    print("TARGET DATA: CONSTRUCTION SPENDING")
    pe_analysis_x_single.run_evaluation("total_construction_spending_health_care.csv", 6)
    print("TARGET DATA: CONSTRUCTION SPENDING - END")

main()
