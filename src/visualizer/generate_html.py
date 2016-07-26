# adapted from https://github.com/steerapi/seq2seq-show-att-tell/blob/master/visualizer/generate_html.py

import os, sys, copy, argparse, shutil, pickle
import numpy as np

def main(arguments):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    html_head_path = os.path.join(script_dir, 'visualizer.html.template.head')
    assert os.path.exists(html_head_path), 'HTML template %s not found'%html_head_path
    html_tail_path = os.path.join(script_dir, 'visualizer.html.template.tail')
    assert os.path.exists(html_tail_path), 'HTML template %s not found'%html_tail_path
    freq_path = os.path.join(script_dir, 'freq.pkl')
    assert os.path.exists(freq_path), 'Freq dict %s not found'%freq_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', dest='output_dir', default='results', help=("Output directory containing results.txt"),
                        type=str)
    parser.add_argument('--data_base_dir', dest='data_base_dir', default='/n/rush_lab/data/image_data/90kDICT32px', help=("The base directory of the image path in data-path. If the image path in data-path is absolute path, set it to /"),
                        type=str)
 
    args = parser.parse_args(arguments)


    output_dir = args.output_dir
    data_base_dir = args.data_base_dir
  
    result_path = os.path.join(output_dir, 'results.txt')
    assert os.path.exists(result_path), 'Result file %s not found'%result_path

    website_dir = os.path.join(output_dir, 'website')
    if not os.path.exists(website_dir):
        os.makedirs(website_dir)

    html_path = os.path.join(website_dir, 'index.html')

    img_dir = os.path.join(website_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    words = pickle.load(open(freq_path))
    with open(result_path) as fin:
        with open(html_head_path) as fhead:
            with open(html_tail_path) as ftail:
                with open(html_path, 'w') as fout:
                    for line in fhead:
                        fout.write(line)
                    fout.write('\n')
                    for line in fin:
                        items = line.strip().split('\t')
                        if len(items) == 5:
                            img_path, label_gold, label_pred, score_pred, score_gold = items
                            freq_gold = words.get(label_gold, 0)
                            freq_pred = words.get(label_pred, 0)
                            img_base_name = img_path.replace('/', '_')[2:]
                            img_path = os.path.join(data_base_dir, img_path)
                            img_new_path = os.path.join(img_dir, img_base_name)
                            shutil.copy(img_path, img_new_path)
                            img_rel_path = os.path.relpath(img_new_path, website_dir)
                            if label_gold == label_pred:
                                s = '<li class="f-correct f-all">\n'
                            else:
                                s = '<li class="f-incorrect f-all">\n'
                            s += '<img src=%s /><br/>\n'%(img_rel_path)
                            s += 'gold: %s (%s)<br/>\n'%(label_gold, score_gold)
                            s += 'predicted: %s (%s)<br/>\n'%(label_pred, score_pred)
                            s += 'gold frequency: %d out of 7.2M<br/>\n'%(freq_gold)
                            s += 'predicted frequency: %d out of 7.2M<br/>\n'%(freq_pred)
                            s += '</li>\n'
                            s += '\n'
                        fout.write(s)


                    for line in ftail:
                        fout.write(line)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
