def log_tensorboard(writer, stats_dict, prefix, step):
    for stat_name, stat_val in stats_dict.items():
        log_path = '%s/%s' % (prefix, stat_name)
        writer.add_scalar(log_path, stat_val, step)


def log_embedding(writer, embed_matrix, prefix, step):
    # TODO, print embedding on tensorboard
    return


def write_args(args):
    params_file = open('%s/params.txt' % args.output_dir, 'w+')
    for attr, value in sorted(args.__dict__.items()):
        params_file.write("%s=%s\n" % (attr, value))
    params_file.close()
