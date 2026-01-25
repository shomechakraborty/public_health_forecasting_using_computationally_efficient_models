import pe_analysis_train_test

def main():
    print("TARGET DATA: EMPLOYEES")
    pe_analysis_train_test.run_evaluation("all_employees_health_care_and_social_assitance.csv", 0.8)
    print("TARGET DATA: EMPLOYEES - END")

main()
