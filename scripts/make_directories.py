import os
dirs_to_create = ['markups_existing',
 'data',
 'nbs',
 'scripts',
 'q_and_a',
 'assets',
 'configs',
 'output_dir',
 'data/markups_existing',
 'data/one_pdf',
 'data/pdfs_ws_mrkp_test',
 'data/abstracts_new_processed',
 'data/markups_new',
 'data/abstracts_new_cant_process',
 'data/markups_new_processed',
 'data/q_and_a',
 'data/ak_articles_up_to',
 'data/rag_index_dir',
 'data/abstracts_existing_cant_process',
 'data/abstracts_existing_processed',
 'data/abstracts',
 'data/markdown',
 'data/abstracts_new',
 'data/one_pdf/pdf_file',
 'data/one_pdf/one_pdf_rag_index',
 'data/one_pdf/one_pdf_rag_index/pdfs',
 'data/pdfs_ws_mrkp_test/mrkps',
 'data/pdfs_ws_mrkp_test/eval_outputs',
 'data/pdfs_ws_mrkp_test/pdfs',
 'data/rag_index_dir/txts',
 'data/rag_index_dir/pdfs']

# Create directories if they don't exist
def make_dirs():
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
if __name__=='__main__':
    make_dirs()
