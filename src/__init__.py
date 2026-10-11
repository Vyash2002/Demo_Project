"""
Predictive maintenance building blocks, one module per pipeline step:

    config        settings
    ingest        1-2. load + clean the 4 CSVs
    validation    data checks (guide 4.5)
    labels        3. failure events and the 7-day target
    features      4. the shared feature function
    sensor_check  stuck / drifting sensor rule
    models        5-7. ML models, calibrated wrapper, SHAP
    training      5-6. time split, CV, calibration, threshold
    evaluation    held-out metrics, slices, error analysis
    report        evaluation report writer
    serving       8. saved model -> per-machine predictions
"""
