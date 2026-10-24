# CHANGELOG

## v0.6.0 [10/24/2026] Automated Model Evaluation and Dashboard Expansion

- Adds automated model evaluation for controled experimentation and model performance/resource consumption of the selected models
- Expands dashboard features, including collapsible model metadata section and model evaluation control from dashboard
- Adds additional tests to test suite to assess new feature additions
- Improves model version control by introducing function to increase version number by 1 from currently existing versions in registry
- Updates and expands documentation

## v0.5.1 [10/19/2026] Test Suite Updates

- Extends test suite to add tests for bounded CSV sampling used by batch training

## v0.5.0 [10/19/2026] Batch Training and Numeric Handling

- Adds batch sampling feature for model training to support memory limitations on training work station
- Fixes numeric handling by adding preprocessing data cleaning for extremely harge or infinity values in dataset
- Updates code organization for unifority and readability
- Extends code comments for better understanding

## v0.4.0 [10/18/2026] Integration and Testing

- Adds system-acceptance testing for API 
- Expands code notes for clarity
- Adds Docker support

## v0.3.0 [10/14/2026] Module Integration

- Adds API feautres to support integrated module performance
- Fully integrates modules
- Adds additional tests to confirm proper integrated behavior
