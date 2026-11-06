"""Unified ATM26 split definitions for Track-1 and Track-2."""

# Shared Shanghai holdout for pipeline-level validation.
# Every case has Track-2 anatomical labels and is excluded from both Track-1 and Track-2 training.
ATM26_SHANGHAI_HOLDOUT = ['Shanghai_031', 'Shanghai_078', 'Shanghai_103', 'Shanghai_120', 'Shanghai_160', 'Shanghai_216', 'Shanghai_221', 'Shanghai_223', 'Shanghai_224', 'Shanghai_226']

list_patient_Shanghai = {'train': ['Shanghai_034',
           'Shanghai_146',
           'Shanghai_199',
           'Shanghai_203',
           'Shanghai_139',
           'Shanghai_117',
           'Shanghai_190',
           'Shanghai_221',
           'Shanghai_017',
           'Shanghai_124',
           'Shanghai_110',
           'Shanghai_218',
           'Shanghai_023',
           'Shanghai_208',
           'Shanghai_046',
           'Shanghai_072',
           'Shanghai_005',
           'Shanghai_084',
           'Shanghai_188',
           'Shanghai_062',
           'Shanghai_053',
           'Shanghai_067',
           'Shanghai_027',
           'Shanghai_168',
           'Shanghai_041',
           'Shanghai_014',
           'Shanghai_123',
           'Shanghai_090',
           'Shanghai_025',
           'Shanghai_031',
           'Shanghai_061',
           'Shanghai_057',
           'Shanghai_163',
           'Shanghai_102',
           'Shanghai_020',
           'Shanghai_170',
           'Shanghai_178',
           'Shanghai_055',
           'Shanghai_182',
           'Shanghai_081',
           'Shanghai_052',
           'Shanghai_109',
           'Shanghai_156',
           'Shanghai_119',
           'Shanghai_087',
           'Shanghai_180',
           'Shanghai_127',
           'Shanghai_161',
           'Shanghai_112',
           'Shanghai_042',
           'Shanghai_015',
           'Shanghai_028',
           'Shanghai_143',
           'Shanghai_205',
           'Shanghai_021',
           'Shanghai_151',
           'Shanghai_047',
           'Shanghai_155',
           'Shanghai_222',
           'Shanghai_131',
           'Shanghai_159',
           'Shanghai_063',
           'Shanghai_148',
           'Shanghai_003',
           'Shanghai_060',
           'Shanghai_138',
           'Shanghai_211',
           'Shanghai_044',
           'Shanghai_011',
           'Shanghai_130',
           'Shanghai_074',
           'Shanghai_217',
           'Shanghai_220',
           'Shanghai_113',
           'Shanghai_202',
           'Shanghai_191',
           'Shanghai_099',
           'Shanghai_004',
           'Shanghai_094',
           'Shanghai_135',
           'Shanghai_207',
           'Shanghai_216',
           'Shanghai_051',
           'Shanghai_114',
           'Shanghai_001',
           'Shanghai_095',
           'Shanghai_096',
           'Shanghai_070',
           'Shanghai_050',
           'Shanghai_049',
           'Shanghai_086',
           'Shanghai_224',
           'Shanghai_024',
           'Shanghai_079',
           'Shanghai_101',
           'Shanghai_132',
           'Shanghai_183',
           'Shanghai_225',
           'Shanghai_007',
           'Shanghai_069',
           'Shanghai_085',
           'Shanghai_122',
           'Shanghai_212',
           'Shanghai_172',
           'Shanghai_192',
           'Shanghai_187',
           'Shanghai_226',
           'Shanghai_092',
           'Shanghai_193',
           'Shanghai_012',
           'Shanghai_120',
           'Shanghai_103',
           'Shanghai_036',
           'Shanghai_058',
           'Shanghai_066',
           'Shanghai_002',
           'Shanghai_121',
           'Shanghai_181',
           'Shanghai_043',
           'Shanghai_106',
           'Shanghai_133',
           'Shanghai_197',
           'Shanghai_018',
           'Shanghai_039',
           'Shanghai_134',
           'Shanghai_054',
           'Shanghai_166',
           'Shanghai_175',
           'Shanghai_129',
           'Shanghai_035',
           'Shanghai_029',
           'Shanghai_115',
           'Shanghai_165',
           'Shanghai_153',
           'Shanghai_032',
           'Shanghai_185',
           'Shanghai_128',
           'Shanghai_200',
           'Shanghai_033',
           'Shanghai_169',
           'Shanghai_144',
           'Shanghai_189',
           'Shanghai_149',
           'Shanghai_030',
           'Shanghai_179',
           'Shanghai_100',
           'Shanghai_083',
           'Shanghai_223',
           'Shanghai_177',
           'Shanghai_080',
           'Shanghai_116',
           'Shanghai_150',
           'Shanghai_219',
           'Shanghai_073',
           'Shanghai_078',
           'Shanghai_026',
           'Shanghai_167',
           'Shanghai_082',
           'Shanghai_213',
           'Shanghai_176',
           'Shanghai_215',
           'Shanghai_040',
           'Shanghai_195',
           'Shanghai_059',
           'Shanghai_141',
           'Shanghai_089',
           'Shanghai_071',
           'Shanghai_088',
           'Shanghai_037',
           'Shanghai_022',
           'Shanghai_214',
           'Shanghai_010',
           'Shanghai_104',
           'Shanghai_198',
           'Shanghai_068',
           'Shanghai_194',
           'Shanghai_118',
           'Shanghai_048',
           'Shanghai_174'],
 'valid': ['Shanghai_209',
           'Shanghai_158',
           'Shanghai_064',
           'Shanghai_016',
           'Shanghai_201',
           'Shanghai_019',
           'Shanghai_206',
           'Shanghai_160',
           'Shanghai_006',
           'Shanghai_173',
           'Shanghai_157',
           'Shanghai_045',
           'Shanghai_076',
           'Shanghai_093',
           'Shanghai_075',
           'Shanghai_091',
           'Shanghai_009',
           'Shanghai_164',
           'Shanghai_154',
           'Shanghai_137',
           'Shanghai_097',
           'Shanghai_204'],
 'test': ['Shanghai_056',
          'Shanghai_008',
          'Shanghai_162',
          'Shanghai_184',
          'Shanghai_108',
          'Shanghai_013',
          'Shanghai_125',
          'Shanghai_038',
          'Shanghai_077',
          'Shanghai_105',
          'Shanghai_136',
          'Shanghai_111',
          'Shanghai_145',
          'Shanghai_171',
          'Shanghai_107',
          'Shanghai_210',
          'Shanghai_186',
          'Shanghai_126',
          'Shanghai_140',
          'Shanghai_098',
          'Shanghai_065',
          'Shanghai_147',
          'Shanghai_152']}

list_patient_LIDC = {'train': ['LIDC_011',
           'LIDC_029',
           'LIDC_023',
           'LIDC_032',
           'LIDC_039',
           'LIDC_038',
           'LIDC_008',
           'LIDC_015',
           'LIDC_028',
           'LIDC_036',
           'LIDC_050',
           'LIDC_019',
           'LIDC_052',
           'LIDC_035',
           'LIDC_016',
           'LIDC_006',
           'LIDC_030',
           'LIDC_017',
           'LIDC_047',
           'LIDC_021',
           'LIDC_053',
           'LIDC_009',
           'LIDC_014',
           'LIDC_026',
           'LIDC_018',
           'LIDC_041',
           'LIDC_044',
           'LIDC_002',
           'LIDC_013',
           'LIDC_042',
           'LIDC_025',
           'LIDC_007',
           'LIDC_024',
           'LIDC_037',
           'LIDC_022',
           'LIDC_020',
           'LIDC_010',
           'LIDC_040',
           'LIDC_051',
           'LIDC_004',
           'LIDC_001',
           'LIDC_048',
           'LIDC_045'],
 'valid': ['LIDC_043', 'LIDC_046', 'LIDC_027', 'LIDC_054', 'LIDC_003'],
 'test': ['LIDC_012', 'LIDC_031', 'LIDC_049', 'LIDC_005', 'LIDC_033', 'LIDC_034']}

list_patient_LIDC_ATM26 = {key:list(value) for key, value in list_patient_LIDC.items()}


list_patient_EXACT09 = {'train': ['EXACT_001', 'EXACT_003', 'EXACT_004', 'EXACT_005', 'EXACT_006', 'EXACT_007', 'EXACT_008', 'EXACT_010', 'EXACT_011', 'EXACT_012',
                                  'EXACT_013', 'EXACT_014', 'EXACT_015', 'EXACT_016', 'EXACT_017'],
 'valid': ['EXACT_002', 'EXACT_019', 'EXACT_018'],
 'test': ['EXACT_009', 'EXACT_020']}

list_patient_BAS = {'train': ['BAS_001', 'BAS_003', 'BAS_004', 'BAS_005', 'BAS_006', 'BAS_007', 'BAS_009', 'BAS_010', 'BAS_011', 'BAS_012',
                              'BAS_013', 'BAS_014', 'BAS_015', 'BAS_016', 'BAS_017', 'BAS_018', 'BAS_019', 'BAS_020', 'BAS_021', 'BAS_022',
                              'BAS_024', 'BAS_025', 'BAS_026', 'BAS_030', 'BAS_033', 'BAS_034', 'BAS_036', 'BAS_037', 'BAS_038', 'BAS_039',
                              'BAS_040', 'BAS_041', 'BAS_042', 'BAS_044', 'BAS_045', 'BAS_046', 'BAS_047', 'BAS_048', 'BAS_050', 'BAS_051',
                              'BAS_053', 'BAS_054', 'BAS_055', 'BAS_056', 'BAS_058', 'BAS_059', 'BAS_061', 'BAS_062', 'BAS_063', 'BAS_064',
                              'BAS_065', 'BAS_066', 'BAS_067', 'BAS_068', 'BAS_069'],
 'valid': ['BAS_023', 'BAS_027', 'BAS_028', 'BAS_032', 'BAS_052', 'BAS_057', 'BAS_060'],
 'test': ['BAS_008', 'BAS_029', 'BAS_031', 'BAS_035', 'BAS_043', 'BAS_049', 'BAS_070']}

list_patient_ATM_Enhanced = {'train': [], 'valid': [], 'test': ['Shanghai_008', 'Shanghai_013', 'Shanghai_038', 'Shanghai_056', 'Shanghai_065', 'Shanghai_077', 'Shanghai_098', 'Shanghai_105', 'Shanghai_107', 'Shanghai_108',
                                       'Shanghai_145', 'Shanghai_147', 'Shanghai_184', 'Shanghai_186', 'Shanghai_210', 'LIDC_005', 'LIDC_031', 'LIDC_033', 'LIDC_034', 'LIDC_049']}

list_patient_Enhanced_Shanghai = [patient for patient in list_patient_ATM_Enhanced['test'] if patient.startswith('Shanghai_')]
list_patient_Enhanced_LIDC = [patient for patient in list_patient_ATM_Enhanced['test'] if patient.startswith('LIDC_')]
for split, patients in [(list_patient_Shanghai, list_patient_Enhanced_Shanghai), (list_patient_LIDC, list_patient_Enhanced_LIDC), (list_patient_LIDC_ATM26, list_patient_Enhanced_LIDC)]:
    split['train'] = [patient for patient in split['train'] if patient not in patients]
    split['valid'] = [patient for patient in split['valid'] if patient not in patients]
    split['test'] = patients.copy()

list_patient_PNUYH = {'train': [],
 'valid': [],
 'test': ['PNUYH_001', 'PNUYH_002', 'PNUYH_003', 'PNUYH_004', 'PNUYH_005', 'PNUYH_006', 'PNUYH_007', 'PNUYH_008', 'PNUYH_009', 'PNUYH_010',
          'PNUYH_011', 'PNUYH_012', 'PNUYH_013', 'PNUYH_014', 'PNUYH_015', 'PNUYH_016', 'PNUYH_017', 'PNUYH_018', 'PNUYH_019', 'PNUYH_020',
          'PNUYH_021', 'PNUYH_022', 'PNUYH_023', 'PNUYH_024', 'PNUYH_025', 'PNUYH_026', 'PNUYH_027', 'PNUYH_029', 'PNUYH_030', 'PNUYH_031',
          'PNUYH_032']}

SPLIT_IDX = {
    'ATM22_Shanghai':list_patient_Shanghai,
    'ATM22_LIDC':list_patient_LIDC,
    'ATM26_Shanghai':list_patient_Shanghai,
    'ATM26_LIDC':list_patient_LIDC_ATM26,
    'ATM26_EXACT09':list_patient_EXACT09,
    'EXACT09':list_patient_EXACT09,
    'BAS':list_patient_BAS,
    'ATM22_Enhanced':list_patient_ATM_Enhanced,
    'ATM26_Enhanced':list_patient_ATM_Enhanced,
    'PNUYH':list_patient_PNUYH}

LIST_PATIENT_T2 = ['Shanghai_001', 'Shanghai_002', 'Shanghai_005', 'Shanghai_006', 'Shanghai_007', 'Shanghai_008', 'Shanghai_009', 'Shanghai_010', 'Shanghai_011', 'Shanghai_012',
                   'Shanghai_014', 'Shanghai_015', 'Shanghai_016', 'Shanghai_017', 'Shanghai_018', 'Shanghai_019', 'Shanghai_021', 'Shanghai_025', 'Shanghai_031', 'Shanghai_032',
                   'Shanghai_039', 'Shanghai_043', 'Shanghai_046', 'Shanghai_050', 'Shanghai_051', 'Shanghai_054', 'Shanghai_055', 'Shanghai_057', 'Shanghai_060', 'Shanghai_061',
                   'Shanghai_062', 'Shanghai_064', 'Shanghai_065', 'Shanghai_066', 'Shanghai_068', 'Shanghai_069', 'Shanghai_073', 'Shanghai_075', 'Shanghai_076', 'Shanghai_078',
                   'Shanghai_079', 'Shanghai_080', 'Shanghai_082', 'Shanghai_083', 'Shanghai_084', 'Shanghai_085', 'Shanghai_089', 'Shanghai_090', 'Shanghai_092', 'Shanghai_093',
                   'Shanghai_096', 'Shanghai_097', 'Shanghai_099', 'Shanghai_103', 'Shanghai_104', 'Shanghai_105', 'Shanghai_107', 'Shanghai_110', 'Shanghai_111', 'Shanghai_112',
                   'Shanghai_117', 'Shanghai_118', 'Shanghai_120', 'Shanghai_123', 'Shanghai_124', 'Shanghai_125', 'Shanghai_126', 'Shanghai_127', 'Shanghai_128', 'Shanghai_129',
                   'Shanghai_132', 'Shanghai_133', 'Shanghai_140', 'Shanghai_141', 'Shanghai_143', 'Shanghai_144', 'Shanghai_145', 'Shanghai_146', 'Shanghai_147', 'Shanghai_150',
                   'Shanghai_151', 'Shanghai_154', 'Shanghai_156', 'Shanghai_157', 'Shanghai_159', 'Shanghai_160', 'Shanghai_166', 'Shanghai_167', 'Shanghai_168', 'Shanghai_172',
                   'Shanghai_173', 'Shanghai_175', 'Shanghai_176', 'Shanghai_177', 'Shanghai_181', 'Shanghai_184', 'Shanghai_185', 'Shanghai_188', 'Shanghai_189', 'Shanghai_191',
                   'Shanghai_192', 'Shanghai_193', 'Shanghai_194', 'Shanghai_197', 'Shanghai_199', 'Shanghai_202', 'Shanghai_204', 'Shanghai_208', 'Shanghai_209', 'Shanghai_212',
                   'Shanghai_213', 'Shanghai_216', 'Shanghai_221', 'Shanghai_223', 'Shanghai_224', 'Shanghai_226', 'LIDC_005', 'EXACT_003']
LIST_PATIENT_SHANGHAI_T2 = [patient for patient in LIST_PATIENT_T2 if patient.startswith('Shanghai_')]
LIST_PATIENT_SHANGHAI_VALID_T2 = ATM26_SHANGHAI_HOLDOUT.copy()
LIST_PATIENT_SHANGHAI_TRAIN_T2 = [patient for patient in LIST_PATIENT_SHANGHAI_T2 if patient not in LIST_PATIENT_SHANGHAI_VALID_T2 and patient not in list_patient_Enhanced_Shanghai]
LIST_PATIENT_SHANGHAI_TEST_T2 = list(dict.fromkeys(LIST_PATIENT_SHANGHAI_VALID_T2+[patient for patient in LIST_PATIENT_SHANGHAI_T2 if patient in list_patient_Enhanced_Shanghai]))
LIST_PATIENT_EXTRA_T2 = [('ATM26_EXACT09', 'EXACT_003')]

SPLIT_IDX_T2 = {
    'ATM26_Shanghai':{'train':LIST_PATIENT_SHANGHAI_TRAIN_T2, 'valid':LIST_PATIENT_SHANGHAI_VALID_T2, 'test':LIST_PATIENT_SHANGHAI_TEST_T2},
    'ATM26_LIDC':{'train':[], 'valid':[], 'test':['LIDC_005']},
    'ATM26_EXACT09':{'train':['EXACT_003'], 'valid':[], 'test':['EXACT_003']}}

def _validate_split_definition():
    t1_train = set(SPLIT_IDX['ATM26_Shanghai']['train'])
    t1_valid = set(SPLIT_IDX['ATM26_Shanghai']['valid'])
    t2_train = set(LIST_PATIENT_SHANGHAI_TRAIN_T2)
    t2_valid = set(LIST_PATIENT_SHANGHAI_VALID_T2)
    if len(ATM26_SHANGHAI_HOLDOUT) != 10: raise ValueError('ATM26 Shanghai holdout must contain exactly 10 cases')
    if t1_train & t1_valid: raise ValueError(f'Track-1 train/valid overlap: {sorted(t1_train & t1_valid)}')
    for name_dset in ['ATM22_Shanghai', 'ATM22_LIDC', 'ATM26_Shanghai', 'ATM26_LIDC']:
        split = SPLIT_IDX[name_dset]
        if any(set(split[a]) & set(split[b]) for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
            raise ValueError(f'{name_dset} splits overlap')
        enhanced_test = set(list_patient_Enhanced_Shanghai if name_dset.endswith('Shanghai') else list_patient_Enhanced_LIDC)
        if set(split['test']) != enhanced_test: raise ValueError(f'{name_dset} test must match Enhanced')
    if t2_train & t2_valid: raise ValueError(f'Track-2 train/valid overlap: {sorted(t2_train & t2_valid)}')
    enhanced = set(list_patient_ATM_Enhanced['test'])
    for name_dset in ['ATM22_Shanghai', 'ATM22_LIDC', 'ATM26_Shanghai', 'ATM26_LIDC']:
        split = SPLIT_IDX[name_dset]
        if enhanced & set(split['train']+split['valid']): raise ValueError(f'Enhanced case leaked into {name_dset} train/valid')
    for name_dset, split in SPLIT_IDX_T2.items():
        if enhanced & set(split['train']+split['valid']): raise ValueError(f'Enhanced case leaked into T2 {name_dset} train/valid')
    if enhanced & {patient for _, patient in LIST_PATIENT_EXTRA_T2}: raise ValueError('Enhanced case leaked into T2 extra train')

_validate_split_definition()

def get_dset_preset(hp, dset_test=None):
    dset_preset = hp.dset_preset
    if dset_test is not None: dset_preset = {'train':hp.dset_preset['train'], 'test':dset_test}

    return dset_preset

def get_split_idx(dset_preset, full_train=False):
    split_idx = {'train':[], 'valid':[], 'test':[]}
    dset_train = dset_preset['train']
    dset_test = list(dset_preset['test'])
    for name_dset in dset_train:
        if name_dset in ['ATM22_Enhanced', 'ATM26_Enhanced'] and name_dset not in dset_test: dset_test.append(name_dset)

    for name_dset in dset_train:
        if name_dset in ['ATM22_Enhanced', 'ATM26_Enhanced']: continue
        patients_train = list(SPLIT_IDX[name_dset]['train'])
        patients_valid = list(SPLIT_IDX[name_dset]['valid'])
        if full_train and name_dset in ['ATM22_Shanghai', 'ATM26_Shanghai']:
            patients_train = list(dict.fromkeys(patients_train+patients_valid))
        split_idx['train'] += [(name_dset, patient) for patient in patients_train]
        split_idx['valid'] += [(name_dset, patient) for patient in patients_valid]

    for name_dset in dset_test:
        if name_dset in dset_train: patients_test = SPLIT_IDX[name_dset]['test']
        else: patients_test = SPLIT_IDX[name_dset]['train']+SPLIT_IDX[name_dset]['valid']+SPLIT_IDX[name_dset]['test']
        split_idx['test'] += [(name_dset, patient) for patient in patients_test]

    return split_idx


def get_split_idx_T2(dset_preset, use_extra_train=False, full_train=False):
    split_idx = {'train':[], 'valid':[], 'test':[]}
    dset_train = dset_preset['train']
    dset_test = dset_preset['test']

    for name_dset in dset_train:
        patients_train = list(SPLIT_IDX_T2[name_dset]['train'])
        patients_valid = list(SPLIT_IDX_T2[name_dset]['valid'])
        if full_train and name_dset == 'ATM26_Shanghai':
            patients_train = list(dict.fromkeys(patients_train+patients_valid))
        split_idx['train'] += [(name_dset, patient) for patient in patients_train]
        split_idx['valid'] += [(name_dset, patient) for patient in patients_valid]

    if use_extra_train:
        for entry in LIST_PATIENT_EXTRA_T2:
            if entry not in split_idx['train']: split_idx['train'].append(entry)

    for name_dset in dset_test:
        if name_dset in dset_train: patients_test = SPLIT_IDX_T2[name_dset]['test']
        else:
            patients_test = SPLIT_IDX_T2[name_dset]['train']+SPLIT_IDX_T2[name_dset]['valid']+SPLIT_IDX_T2[name_dset]['test']
            patients_test = list(dict.fromkeys(patients_test))
        split_idx['test'] += [(name_dset, patient) for patient in patients_test]

    return split_idx
