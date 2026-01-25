import pe_analysis_train_test

def main():
    print("TARGET DATA: BUSINESS APPLICATIONS")
    pe_analysis_train_test.run_evaluation("business_applications_health_care_and_social_assistance.csv", 0.8)
    print("TARGET DATA: BUSINESS APPLICATIONS - END")

main()
