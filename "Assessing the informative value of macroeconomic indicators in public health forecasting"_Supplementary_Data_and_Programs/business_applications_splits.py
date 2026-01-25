import pe_analysis_splits

def main():
    print("TARGET DATA: BUSINESS APPLICATIONS")
    pe_analysis_splits.run_evaluation("business_applications_health_care_and_social_assistance.csv", 12, 4)
    print("TARGET DATA: BUSINESS APPLICATIONS - END")

main()
