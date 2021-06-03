kiara run examples/pipelines/topic_modeling_create_table.json path=/home/markus/projects/dharpa/notebooks/TopicModelling/data_tm_workflow



kiara run onboarding.import_local_folder path=/home/markus/projects/dharpa/notebooks/TopicModelling/data_tm_workflow/ --save --id topic_modelling_files

kiara run tabular.create_table_from_text_files files=value:topic_modelling_files.file_bundle --save --id topic_modeling_table -o format=silent

kiara run tabular.cut_column table=value:topic_modeling_table.table column_name=rel_path --save --id date_column
