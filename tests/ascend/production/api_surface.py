def assert_no_testing_diagnostic_surface(extension):
    for subject_name, subject in (
            ("module", extension),
            ("ElasticBuffer", extension.ElasticBuffer)):
        for attribute in dir(subject):
            normalized = attribute.lower()
            if "testing" in normalized or "diagnostic" in normalized:
                raise AssertionError(
                    f"{subject_name}.{attribute} exposes test-only state")
