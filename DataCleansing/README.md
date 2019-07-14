## Crate links to the dataset
ln -s m2cai_dataset_path/ m2cai16-workflow-5
ln -s cholec_dataset_path/ cholec80-workflow-5

## Cleanse training set with cross-validation
```
python main.py --dataset=m2cai16-workflow-5 --cleansing=workflow_video_01#workflow_video_02 --savepath=cleansing_one_fold -ground_truth_path=annotation_folder
```
The example command will do the datacleansing for workflow_video_01 and workflow_video_02, the results will be saved in --savepath. After you finished cleansing for all training videos, put the result .txt file into a folder named 'resnet', and put it into the dir : 'm2cai16-workflow-5/train_dataset/'. But this is time-costing, we suggest you skip this step, and use the cleansing results we provided directly.

## The performance gain after removing hard frames.
```
python resnet_baseline.py --dataset=m2cai16-workflow-5 --savepath=m2cai_gain --ground_truth_path=annotaion_folder
```
You will get the final results at --savepth.
