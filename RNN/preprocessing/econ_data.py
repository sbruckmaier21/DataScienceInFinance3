from cif import cif


def load_and_preprocess_econ_data():
    data, subjects, measures = cif.createDataFrameFromOECD(countries=['USA'], dsname='MEI', frequency='M')
    econ_data = data['USA'][
        ['XTNTVA01', 'PRMNTO01', 'IRLTCT01', 'CPALCY01', 'SLMNTO02', 'CPALTT01', 'LOCOBSNO',
         'LOCOCINO', 'LOCODWNO', 'LOCOSINO', 'LORSGPNO', 'LRUNTTTT', 'LMJVTTUV']]
    econ_data = econ_data.reset_index()
    econ_data = econ_data.loc[(econ_data['time'] >= '2012-01') & (econ_data['time'] <= '2022-03')]
    econ_data = econ_data.drop(['CXMLSA', 'NCMLSA', 'IXOBSA', 'NCML', 'MLSA', 'CTGY', 'IXNB', 'IXOB'], axis=1, level=1)
    econ_data.columns = econ_data.columns.map('_'.join).str.strip('_')
    econ_data = econ_data.drop(columns=['LMJVTTUV_STSA', 'LRUNTTTT_STSA'])
    econ_data.to_csv('training_data/econ_data_oecd.csv', index=False)
    return econ_data