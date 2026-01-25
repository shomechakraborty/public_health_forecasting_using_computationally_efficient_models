import pe_analysis_splits

def main():
    print("TARGET DATA: EMPLOYEES")
    pe_analysis_splits.run_evaluation("all_employees_health_care_and_social_assitance.csv", 12, 4)
    print("TARGET DATA: EMPLOYEES - END")

main()
