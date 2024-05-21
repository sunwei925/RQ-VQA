1) To set up the envronment for q-align
use the command: 

pip install -e .


2) extract the Q-align features:

python -u q_align_feature_extract.py --input_path "path-to-the-frames" --output_prefix "path-to-the-saved-features"

A). The input path is the "path to the frames". The structure is like:

input_path|---0001|----000.png
	             |----001.png
                            ....
                |---0002 ....

B). The ouput prefix is the path you want to save the q-align features.


3) The code automatically download the q-align model weight from the huggingface, which is about 20G.
If you have the weight locally, change the "using_local_model = False" on line 41 of "q_align_feature_extract.py"
and update the weight path on line 48.