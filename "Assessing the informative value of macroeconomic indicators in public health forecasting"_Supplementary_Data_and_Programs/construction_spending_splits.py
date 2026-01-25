import pe_analysis_splits

def main():
    print("TARGET DATA: CONSTRUCTION SPENDING")
    pe_analysis_splits.run_evaluation("total_construction_spending_health_care.csv", 12, 4)
    print("TARGET DATA: CONSTRUCTION SPENDING - END")

main()
