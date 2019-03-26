from data_processing import labelling_data as lbld

if __name__ == '__main__':
    # lbld.prepare_annotation("/home/mathias/Documents/Datasets/BeachVolleyball/08032019_150000/")
    # lbld.create_labelling_project_files("/home/mathias/Documents/Datasets/BeachVolleyball/20022019_110000/raw/")
    # lbld.create_merge_file("/home/mathias/Documents/Datasets/BeachVolleyball/20022019_110000/")
    lbld.process_data_3dmodel("/home/mathias/Documents/Datasets/BeachVolleyball/08032019_150000/")
