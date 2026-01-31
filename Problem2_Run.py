from dwts_analysis_clean import DWTSPipeline

pipeline = DWTSPipeline(csv_path="2026_MCM_Problem_C_Data.csv", out_dir="dwts_outputs")
pipeline.run_for_season(2) 
pipeline.run_for_season(11)
pipeline.run_for_season(27)