import pe_analysis_train_test

def main():
    print("TARGET DATA: CONSTRUCTION SPENDING")
    pe_analysis_train_test.run_evaluation("total_construction_spending_health_care.csv", 0.8)
    print("TARGET DATA: CONSTRUCTION SPENDING - END")

main()
