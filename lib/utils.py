def write_result_to_txt(args,results):
    file_name=args.output_dir+args.results_file
    with open(file_name,"a") as file:
        file.write("output_dir:"+args.output_dir+"\n")
        file.write("source_domains:"+str(args.source_domains)+"\n")
        file.write("target_domains:"+str(args.target_domains)+"\n")
        file.write("n_train:"+str(results[0])+"\n")
        file.write("n_test:"+str(results[1])+"\n")
        file.write("best_acc:"+str(results[2])+"\n")
        file.write("\n")