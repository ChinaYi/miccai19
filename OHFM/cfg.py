'''
    Some statistical results for both datasets.
    Including mean, std, so on.

'''
config = {}
config['cholec80-workflow-5'] = {}
config['m2cai16-workflow-5'] = {}

config['m2cai16-workflow-5']['mean'] = [0.4046354902199179, 0.28180467703284984, 0.27697674878095108]
config['m2cai16-workflow-5']['std'] = [0.20434530256293329, 0.17285698229209737, 0.16687120815779266]
config['cholec80-workflow-5']['mean'] = [0.40521515692759497, 0.27927462047480039, 0.27426218748099274]
config['cholec80-workflow-5']['std'] = [0.20460533490722591, 0.17244239120062696, 0.16623196974782356]

config['m2cai16-workflow-5']['phase_name'] = ['TrocarPlacement','Preparation','CalotTriangleDissection','ClippingCutting',
                'GallbladderDissection','GallbladderPackaging','CleaningCoagulation','GallbladderRetraction']
config['m2cai16-workflow-5']['mapping_dict'] = {
    'TrocarPlacement': 0, 'Preparation':1,
    'CalotTriangleDissection':2, 'ClippingCutting':3,
    'GallbladderDissection':4,'GallbladderPackaging':5,
    'CleaningCoagulation':6,'GallbladderRetraction':7,
    'Negative':8
}
config['cholec80-workflow-5']['phase_name'] = ['Preparation','CalotTriangleDissection','ClippingCutting',
                'GallbladderDissection','GallbladderPackaging','CleaningCoagulation','GallbladderRetraction']
config['cholec80-workflow-5']['mapping_dict'] = {
    'Preparation':0, 'CalotTriangleDissection':1,
    'ClippingCutting':2, 'GallbladderDissection':3,
    'GallbladderPackaging':4,'CleaningCoagulation':5,
    'GallbladderRetraction':6, 'Negative':7
}

config['m2cai16-workflow-5']['reverse_mapping_dict'] = {config['m2cai16-workflow-5']['mapping_dict'][key] : key for key in config['m2cai16-workflow-5']['mapping_dict'].keys()}
config['cholec80-workflow-5']['reverse_mapping_dict'] = {config['cholec80-workflow-5']['mapping_dict'][key] : key for key in config['cholec80-workflow-5']['mapping_dict'].keys()}

# # Possible transitions by static information
# config['m2cai16-workflow-5']['transition'] = [
#     {2,3,4,5,6,7},
#     {3,4,5,6,7},
#     {0,4,5,6,7},
#     {0,1,5,6,7},
#     {0,1,2,3,7},
#     {0,1,2,3,4},
#     {0,1,2,3,4},
#     {0,1,2,3,4}
# ]
# config['m2cai16-workflow-5']['previous_phase'] = [0,1,2,3,4,5,6,7]
# config['m2cai16-workflow-5']['max_start'] = 10000 # Prepration must end
# config['m2cai16-workflow-5']['phase_nums'] = [100,100,100,100,100,100,100,100] # useless, the avg length for each phase, but I did not take it into count.
#
# config['cholec80-workflow-5']['transition'] = [
#     {2,3,4,5,6},
#     {3,4,5,6},
#     {0,1,4,5,6},
#     {0,1,6},
#     {0,1,2,3},
#     {0,1,2,3},
#     {0,1,2,3,4},
# ]
# config['cholec80-workflow-5']['previous_phase'] = [1,2,4,5]
# config['cholec80-workflow-5']['max_start'] = 1000
# config['cholec80-workflow-5']['phase_nums'] = [100,100,100,100,3,100,100] # useless

config['m2cai16-workflow-5']['trainset_len'] = {'workflow_video_06': 73411, 'workflow_video_12': 86140, 'workflow_video_19': 73991, 'workflow_video_23': 51114, 'workflow_video_01': 76426, 'workflow_video_02': 21838, 'workflow_video_03': 80375, 'workflow_video_04': 92033, 'workflow_video_05': 51426, 'workflow_video_20': 50650, 'workflow_video_21': 68950, 'workflow_video_22': 64358, 'workflow_video_13': 40075, 'workflow_video_14': 81090, 'workflow_video_15': 68015, 'workflow_video_16': 77347, 'workflow_video_17': 76376, 'workflow_video_18': 42501, 'workflow_video_07': 37476, 'workflow_video_08': 45571, 'workflow_video_09': 54543, 'workflow_video_10': 38637, 'workflow_video_11': 53312, 'workflow_video_24': 99854, 'workflow_video_25': 74660, 'workflow_video_26': 62519, 'workflow_video_27': 46550}
config['m2cai16-workflow-5']['testset_len'] = {'test_workflow_video_05': 29401, 'test_workflow_video_06': 75176, 'test_workflow_video_07': 38012, 'test_workflow_video_10': 82870, 'test_workflow_video_11': 30001, 'test_workflow_video_02': 45220, 'test_workflow_video_04': 35051, 'test_workflow_video_01': 29001, 'test_workflow_video_08': 70907, 'test_workflow_video_09': 75165, 'test_workflow_video_03': 17751, 'test_workflow_video_12': 61295, 'test_workflow_video_14': 36203, 'test_workflow_video_13': 47878}

config['cholec80-workflow-5']['trainset_len'] = {'video01': 43327, 'video02': 70977, 'video03': 145702, 'video04': 38052, 'video05': 58602, 'video06': 53827, 'video07': 113927, 'video08': 37977, 'video10': 43727, 'video11': 80502, 'video12': 27252, 'video13': 24527, 'video14': 42702, 'video15': 51453, 'video16': 73927, 'video17': 32602, 'video19': 60602, 'video20': 36227, 'video21': 31452, 'video22': 38302, 'video23': 40877, 'video24': 49377, 'video25': 53227, 'video26': 44327, 'video28': 29977, 'video29': 58752, 'video30': 73127, 'video31': 98627, 'video32': 52902, 'video34': 33077, 'video35': 52652, 'video36': 59677, 'video37': 30803, 'video38': 77002, 'video39': 41177, 'video40': 55552, 'video09': 67552, 'video18': 48552, 'video27': 52102, 'video33': 32677}
config['cholec80-workflow-5']['testset_len'] = {'video41': 77577, 'video42': 92802, 'video43': 59052, 'video50': 27352, 'video51': 73602, 'video52': 49152, 'video59': 26152, 'video60': 63302, 'video61': 110227, 'video68': 49302, 'video69': 114377, 'video74': 40852, 'video75': 48077, 'video46': 41327, 'video49': 41777, 'video55': 25927, 'video58': 149827, 'video64': 59952, 'video67': 58827, 'video73': 33902, 'video44': 78177, 'video45': 84677, 'video47': 56477, 'video48': 45852, 'video53': 82077, 'video54': 77502, 'video56': 45852, 'video57': 65802, 'video62': 50802, 'video63': 85827, 'video65': 46827, 'video66': 45602, 'video70': 29852, 'video71': 62877, 'video72': 77652, 'video76': 66227, 'video77': 62552, 'video78': 18477, 'video79': 85352, 'video80': 43102}

