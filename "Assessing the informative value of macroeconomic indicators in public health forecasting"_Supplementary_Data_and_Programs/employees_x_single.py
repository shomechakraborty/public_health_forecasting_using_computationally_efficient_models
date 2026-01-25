import pe_analysis_x_single

def main():
    print("TARGET DATA: EMPLOYEES")
    pe_analysis_x_single.run_evaluation("all_employees_health_care_and_social_assitance.csv", 6)
    print("TARGET DATA: EMPLOYEES - END")

main()
