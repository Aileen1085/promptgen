import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import nibabel as nib


def module():
    path = Path(__file__).resolve().parents[1] / 'tools/four_source_data.py'
    assert path.is_file(), 'independent native four-source reader is missing'
    spec = importlib.util.spec_from_file_location('four_source_data', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selection_is_reproducible_and_independent_of_input_order():
    m = module()
    rows = [dict(case_id=f'{i:03}', source_name='fumpe') for i in range(35)]
    picked = m.select_cases(rows)
    assert len(picked) == len({r['case_id'] for r in picked}) == 30
    assert picked == m.select_cases(list(reversed(rows)))
    assert picked != m.select_cases(rows, seed=123)


def test_selection_rejects_insufficient_or_duplicate_cases():
    m = module()
    with pytest.raises(ValueError, match='insufficient'):
        m.select_cases([dict(case_id='a')])
    with pytest.raises(ValueError, match='duplicate'):
        m.select_cases([dict(case_id='a')] * 30)


def test_pengwin_merges_instance_ranges_and_rejects_invalid_labels():
    m = module()
    assert m.merge_pengwin_labels(np.array([0,1,50,51,100,101,150,151,200])).tolist() == [0,1,1,2,2,3,3,4,4]
    for a in [np.array([201]), np.array([-1]), np.array([1.5]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            m.merge_pengwin_labels(a)


def test_pair_canonicalizes_flipped_physical_grid_without_ct_normalization(tmp_path):
    m = module()
    ct = np.arange(24, dtype=np.float32).reshape(2,3,4) - 1000
    mask = (ct > -990).astype(np.uint8)
    affine = np.diag([-2.,3.,4.,1.]); affine[0,3] = 2.
    image_path = tmp_path/'image.nii.gz'; label_path = tmp_path/'label.nii.gz'
    nib.save(nib.Nifti1Image(ct, affine), image_path)
    # Label may have its own equivalent stored orientation.
    ras = affine.copy(); ras[0,0] = 2.; ras[0,3] = 0.
    nib.save(nib.Nifti1Image(mask[::-1], ras), label_path)
    image, label, audit = m.load_pair(dict(source_name='fumpe',case_id='a',image_path=str(image_path),label_path=str(label_path)))
    assert nib.aff2axcodes(image.affine) == ('R','A','S')
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), ct[::-1])
    np.testing.assert_array_equal(np.asanyarray(label.dataobj), mask[::-1])
    assert audit['grid_verified'] and audit['ct_normalization'] == 'none'
    assert audit['original_image_axis_codes'] == ['L','A','S']
    assert audit['original_label_axis_codes'] == ['R','A','S']
    assert audit['image_to_ras_orientation_transform'][0] == [0.,-1.]
    assert audit['source_signatures'][0]['size_bytes'] == image_path.stat().st_size
    assert 'qform_sform_disagreement' in audit['original_image_header']


def test_pair_rejects_subvoxel_affine_shift(tmp_path):
    m = module(); a = np.zeros((2,3,4), dtype=np.float32)
    image = tmp_path/'image.nii.gz'; label = tmp_path/'label.nii.gz'
    nib.save(nib.Nifti1Image(a, np.eye(4)), image)
    affine = np.eye(4); affine[0,3] = .01
    nib.save(nib.Nifti1Image(a, affine), label)
    with pytest.raises(ValueError, match='grid|geometry'):
        m.load_pair(dict(source_name='fumpe',case_id='a',image_path=str(image),label_path=str(label)))


def test_pair_rejects_singular_or_nonfinite_affine():
    m = module()
    for invalid in (0., float('nan')):
        image = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.float32), np.eye(4))
        label = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.uint8), np.eye(4))
        image.affine[0,0] = invalid; label.affine[0,0] = invalid
        with pytest.raises(ValueError,match='affine'):
            m._canonical_pair(image,label,'fumpe')


def ct_slice(z, uid=None, series='ct'):
    return NS(SOPInstanceUID=uid or f'ct{z}', SeriesInstanceUID=series,
              FrameOfReferenceUID='frame', ImageOrientationPatient=[1,0,0,0,1,0],
              ImagePositionPatient=[0,0,z], PixelSpacing=[2,3],
              Rows=2, Columns=3, RescaleSlope=2, RescaleIntercept=-1000,
              pixel_array=np.full((2,3),z, dtype=np.int16))


def seg_frame(z, uid=None, x=0):
    return NS(PlanePositionSequence=[NS(ImagePositionPatient=[x,0,z])],
              SegmentIdentificationSequence=[NS(ReferencedSegmentNumber=1)],
              DerivationImageSequence=[NS(SourceImageSequence=[NS(ReferencedSOPInstanceUID=uid or f'ct{z}')])])


def seg(frames=None):
    frames = frames or [seg_frame(2), seg_frame(0)]
    return NS(FrameOfReferenceUID='frame', SegmentationType='BINARY',
              SegmentSequence=[NS(SegmentNumber=1, SegmentLabel='ACC tumor')],
              ReferencedSeriesSequence=[NS(SeriesInstanceUID='ct')],
              SharedFunctionalGroupsSequence=[NS(
                  PlaneOrientationSequence=[NS(ImageOrientationPatient=[1,0,0,0,1,0])],
                  PixelMeasuresSequence=[NS(PixelSpacing=[2,3])])],
              PerFrameFunctionalGroupsSequence=frames,
              pixel_array=np.ones((len(frames),2,3), dtype=np.uint8))


def test_dicom_uses_physical_positions_rescale_and_seg_sop_references():
    m = module()
    image, label, audit = m.dicom_pair_from_datasets([ct_slice(2),ct_slice(0),ct_slice(1)], seg())
    assert image.shape == label.shape == (3,2,3)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj)[0,0], [-1000,-998,-996])
    np.testing.assert_array_equal(np.asanyarray(label.dataobj)[0,0], [1,0,1])
    assert audit['seg_frames_placed'] == 2


@pytest.mark.parametrize('kind', ['shift','sop','frame','series','duplicate','unrepresentable'])
def test_dicom_rejects_inconsistent_reference_or_geometry(kind):
    m = module(); slices = [ct_slice(0),ct_slice(1),ct_slice(2)]; s = seg()
    if kind == 'shift': s.PerFrameFunctionalGroupsSequence[0] = seg_frame(2,x=.01)
    if kind == 'sop': s.PerFrameFunctionalGroupsSequence[0] = seg_frame(2,uid='absent')
    if kind == 'frame': s.FrameOfReferenceUID = 'other'
    if kind == 'series': s.ReferencedSeriesSequence[0].SeriesInstanceUID = 'other'
    if kind == 'duplicate': slices[2] = ct_slice(1,uid='duplicate')
    if kind == 'unrepresentable': slices[2] = ct_slice(1.1234567)
    with pytest.raises(ValueError): m.dicom_pair_from_datasets(slices, s)


def test_nonuniform_dicom_reconstructs_exact_quarter_mm_lattice():
    m = module()
    native = [ct_slice(0,uid='ct0'),ct_slice(.5,uid='ct1'),ct_slice(1.75,uid='ct2')]
    for image,z in zip(native,[0.,.5,1.75]):
        image.pixel_array = np.full((2,3),z,dtype=np.float32)
    segmentation = seg([seg_frame(0,uid='ct0'),seg_frame(1.75,uid='ct2')])
    segmentation.pixel_array = np.array([np.zeros((2,3)),np.ones((2,3))],dtype=np.uint8)
    image,label,audit = m.dicom_pair_from_datasets(native,segmentation)
    np.testing.assert_allclose(nib.affines.voxel_sizes(image.affine),[3,2,.25])
    assert image.shape == label.shape == (3,2,8)
    np.testing.assert_array_equal(np.asarray(image.dataobj)[0,0],np.arange(8)*.5-1000)
    np.testing.assert_array_equal(np.asarray(label.dataobj)[0,0],[0,0,0,0,0,1,1,1])
    regularization = audit['z_regularization']
    assert regularization['applied'] and regularization['lattice_pitch_mm'] == .25
    assert regularization['native_to_lattice_indices'] == [0,2,7]
    assert regularization['native_z_pitch_range_mm'] == [.5,1.25]
    assert regularization['native_ipp_lps_mm'] == [[0,0,0],[0,0,.5],[0,0,1.75]]
    assert regularization['new_depth'] == 8
    assert 'linear' in regularization['ct_interpolation']
    assert 'nearest' in regularization['label_interpolation']
    for native_z,lattice_z in enumerate([0,2,7]):
        np.testing.assert_array_equal(np.asarray(image.dataobj)[:,:,lattice_z],native[native_z].pixel_array.T*2-1000)


def test_uniform_dicom_does_not_interpolate_or_regularize():
    m = module()
    _,_,audit = m.dicom_pair_from_datasets([ct_slice(0),ct_slice(1),ct_slice(2)],seg())
    assert not audit['z_regularization']['applied']


@pytest.mark.parametrize('positions', [[0,.1,.3],[0,.25,2000]])
def test_nonuniform_dicom_rejects_too_fine_or_excessive_lattice(positions):
    m = module()
    slices = [ct_slice(z,uid=f'ct{i}') for i,z in enumerate(positions)]
    s = seg([seg_frame(positions[0],uid='ct0'),seg_frame(positions[-1],uid='ct2')])
    with pytest.raises(ValueError,match='lattice'):
        m.dicom_pair_from_datasets(slices,s)


def test_nonuniform_dicom_still_rejects_true_seg_position_mismatch():
    m = module()
    slices = [ct_slice(0,uid='ct0'),ct_slice(.5,uid='ct1'),ct_slice(1.75,uid='ct2')]
    s = seg([seg_frame(0,uid='ct0'),seg_frame(1.76,uid='ct2')])
    with pytest.raises(ValueError,match='physical plane'):
        m.dicom_pair_from_datasets(slices,s)


def test_nearly_uniform_gaps_cannot_accumulate_affine_position_error():
    m = module()
    positions = [0,1,2.00009,3.00018,4.00027]
    slices = [ct_slice(z,uid=f'ct{i}') for i,z in enumerate(positions)]
    s = seg([seg_frame(positions[0],uid='ct0'),seg_frame(positions[-1],uid='ct4')])
    with pytest.raises(ValueError,match='lattice|uniform'):
        m.dicom_pair_from_datasets(slices,s)


def test_rounded_oblique_iop_uses_unit_normal_for_exact_long_z_lattice():
    m = module()
    orientation = [.707107,.707107,0,-.707107,.707107,0]
    slices = [ct_slice(z,uid=f'ct{i}') for i,z in enumerate([0,.5,399.75])]
    for image in slices: image.ImageOrientationPatient = orientation
    s = seg([seg_frame(0,uid='ct0'),seg_frame(399.75,uid='ct2')])
    s.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient = orientation
    image,_,audit = m.dicom_pair_from_datasets(slices,s)
    assert image.shape[2] == 1600
    assert audit['z_regularization']['lattice_pitch_mm'] == .25
    assert audit['z_regularization']['native_to_lattice_indices'] == [0,2,1599]
    np.testing.assert_allclose(np.linalg.norm(image.affine[:3,2]),.25,rtol=0,atol=1e-12)


def test_zero_seg_frames_allow_narrow_audited_malformed_reference_handling():
    m = module()
    s = seg([seg_frame(1),seg_frame(3)])
    s.pixel_array[:] = 0
    s.PerFrameFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence = [
        NS(ReferencedSOPInstanceUID='ct0'),NS(ReferencedSOPInstanceUID='ct1')]
    s.PerFrameFunctionalGroupsSequence[1].DerivationImageSequence = []
    _,label,audit = m.dicom_pair_from_datasets([ct_slice(0),ct_slice(1),ct_slice(2)],s)
    assert not np.any(label.dataobj)
    assert audit['zero_frames_resolved_by_unique_ipp'] == 1
    assert audit['zero_frames_omitted_missing_sop_outside_ct'] == 1
    assert audit['seg_frames_placed'] == 1 and audit['seg_frames_total'] == 2
    resolved,omitted = audit['zero_frame_reference_exceptions']
    assert resolved['frame_index'] == 0 and resolved['resolved_sop_uid'] == 'ct1'
    assert resolved['ipp_lps_mm'] == [0.,0.,1.] and resolved['pixel_sum'] == 0
    assert omitted['frame_index'] == 1 and omitted['ipp_lps_mm'] == [0.,0.,3.]
    assert omitted['action'] == 'omit' and omitted['pixel_sum'] == 0


@pytest.mark.parametrize('kind',['positive_multiple','positive_missing','zero_unknown','zero_multiple_no_ipp',
                                 'zero_multiple_unreferenced_ipp','zero_missing_existing_ipp',
                                 'zero_missing_inside_gap','zero_missing_outside_xy','wrong_frame'])
def test_zero_reference_exceptions_do_not_accept_other_reference_failures(kind):
    m = module(); s = seg([seg_frame(1)]); s.pixel_array[:] = 0
    frame = s.PerFrameFunctionalGroupsSequence[0]
    frame.DerivationImageSequence[0].SourceImageSequence = [NS(ReferencedSOPInstanceUID='ct0'),NS(ReferencedSOPInstanceUID='ct1')]
    if kind == 'positive_multiple': s.pixel_array[:] = 1
    if kind == 'positive_missing': s.pixel_array[:] = 1; frame.DerivationImageSequence = []
    if kind == 'zero_unknown': frame.DerivationImageSequence[0].SourceImageSequence.append(NS(ReferencedSOPInstanceUID='absent'))
    if kind == 'zero_multiple_no_ipp': frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,3]
    if kind == 'zero_multiple_unreferenced_ipp': frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,2]
    if kind == 'zero_missing_existing_ipp': frame.DerivationImageSequence = []
    if kind == 'zero_missing_inside_gap': frame.DerivationImageSequence = []; frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,.5]
    if kind == 'zero_missing_outside_xy': frame.DerivationImageSequence = []; frame.PlanePositionSequence[0].ImagePositionPatient = [.01,0,3]
    if kind == 'wrong_frame': s.FrameOfReferenceUID = 'other'; frame.DerivationImageSequence = []; frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,3]
    with pytest.raises(ValueError):
        m.dicom_pair_from_datasets([ct_slice(0),ct_slice(1),ct_slice(2)],s)


def independent_seg_fixture():
    slices = [ct_slice(z,uid=f'ct{i}') for i,z in enumerate([0,.5,2.5,4.5])]
    s = seg([seg_frame(0,uid='ct0'),seg_frame(2,uid='ct2'),seg_frame(4,uid='ct3')])
    s.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = 2
    s.pixel_array[:] = 0; s.pixel_array[1] = 1
    return slices,s


def test_independent_regular_seg_preserves_physical_offset_instead_of_sop_snap():
    m = module(); slices,s = independent_seg_fixture()
    image,label,audit = m.dicom_pair_from_datasets(slices,s)
    assert image.shape == label.shape == (3,2,10)
    np.testing.assert_array_equal(np.asarray(label.dataobj)[0,0],[0,0,0,1,1,1,1,0,0,0])
    info = audit['independent_seg_grid']
    assert info['applied'] and info['seg_native_pitch_mm'] == 2
    assert info['seg_reference_ipp_offset_range_mm'] == [-.5,0.]
    assert info['positive_seg_planes_within_ct_span']
    assert info['positive_seg_to_ct_lattice_indices'] == [4]
    assert not info['sop_references_used_for_placement']
    assert info['seg_native_ipp_lps_mm'] == [[0,0,0],[0,0,2],[0,0,4]]


@pytest.mark.parametrize('kind',['declared_pitch','xy','frame','positive_outside','positive_missing_ref','unknown_ref','lattice'])
def test_independent_seg_rejects_invalid_geometry_or_references(kind):
    m = module(); slices,s = independent_seg_fixture()
    if kind == 'declared_pitch': s.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = 3
    if kind == 'xy':
        for frame in s.PerFrameFunctionalGroupsSequence: frame.PlanePositionSequence[0].ImagePositionPatient[0] = .01
    if kind == 'frame': s.FrameOfReferenceUID = 'other'
    if kind == 'positive_outside':
        s.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = 3
        for frame,z in zip(s.PerFrameFunctionalGroupsSequence,[0,3,6]): frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,z]
        s.pixel_array[2] = 1
    if kind == 'positive_missing_ref': s.PerFrameFunctionalGroupsSequence[1].DerivationImageSequence = []
    if kind == 'unknown_ref': s.PerFrameFunctionalGroupsSequence[1].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID = 'absent'
    if kind == 'lattice':
        slices = [ct_slice(z,uid=f'ct{i}') for i,z in enumerate([0,1,3,5])]
        s.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = 1.5
        for frame,z in zip(s.PerFrameFunctionalGroupsSequence,[0,1.5,3]): frame.PlanePositionSequence[0].ImagePositionPatient = [0,0,z]
    with pytest.raises(ValueError): m.dicom_pair_from_datasets(slices,s)


def test_discover_fumpe_and_graz_records_are_json_safe(tmp_path):
    import json
    m = module()
    for p in ['FUMPE/images/PAT001.nii.gz','FUMPE/masks/PAT001.nii.gz',
              'Aortic-Dissection-Graz/dataset/cta01s/cta01s.nrrd',
              'Aortic-Dissection-Graz/dataset/cta01s/mask01.nrrd']:
        path = tmp_path/p; path.parent.mkdir(parents=True,exist_ok=True); path.touch()
    found = m.discover(tmp_path)
    assert found['fumpe'][0]['case_id'] == 'PAT001'
    assert found['graz'][0]['case_id'] == 'cta01s'
    json.dumps(found)


def test_embedded_gzipped_nifti_preserves_hu_and_geometry():
    import gzip
    m = module()
    values = np.arange(24, dtype=np.float32).reshape(2,3,4)-1000
    original = nib.Nifti1Image(values, np.diag([2.,3.,4.,1.]))
    restored = m._embedded_nifti(gzip.compress(original.to_bytes()))
    np.testing.assert_array_equal(restored.dataobj, values)
    np.testing.assert_array_equal(restored.affine, original.affine)


def test_embedded_nifti_preserves_original_forms_without_double_hu_scaling():
    m = module()
    stored = np.arange(24,dtype=np.int16).reshape(2,3,4)
    original = nib.Nifti1Image(stored,np.eye(4))
    qform = np.eye(4); qform[0,3] = 10
    original.set_qform(qform,code=1)
    original.set_sform(np.eye(4),code=2)
    original.header.set_slope_inter(2.,-1000.)
    restored = m._embedded_nifti(original.to_bytes())
    np.testing.assert_array_equal(restored.dataobj,stored*2.-1000.)
    q, qcode = restored.get_qform(coded=True)
    s, scode = restored.get_sform(coded=True)
    assert int(qcode) == 1 and int(scode) == 2
    np.testing.assert_array_equal(q,qform)
    np.testing.assert_array_equal(s,np.eye(4))
    label = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.uint8),np.eye(4))
    image, _, audit = m._canonical_pair(restored,label,'pengwin')
    assert audit['original_image_header']['qform_sform_disagreement']
    np.testing.assert_array_equal(image.dataobj,stored*2.-1000.)


def test_parquet_discovery_and_native_load(tmp_path):
    import duckdb
    m = module()
    p = tmp_path/'PENGWIN2026/data/train-00000-of-00020.parquet'; p.parent.mkdir(parents=True)
    ct = nib.Nifti1Image(np.arange(24,dtype=np.float32).reshape(2,3,4)-1000, np.eye(4))
    label = nib.Nifti1Image(np.full((2,3,4),51,dtype=np.uint8), np.eye(4))
    c = duckdb.connect()
    try:
        c.execute('CREATE TABLE pairs(image STRUCT(bytes BLOB,path VARCHAR),label STRUCT(bytes BLOB,path VARCHAR))')
        c.execute('INSERT INTO pairs VALUES (?,?)', [{'bytes':ct.to_bytes(),'path':'001_image.nii.gz'},
                                                     {'bytes':label.to_bytes(),'path':'001_label.nii.gz'}])
        c.execute('COPY pairs TO ? (FORMAT PARQUET)', [str(p)])
    finally: c.close()
    record = m.discover(tmp_path)['pengwin'][0]
    assert record['case_id'] == '001' and record['row_index'] == 0
    image, merged, _ = m.load_pair(record)
    np.testing.assert_array_equal(image.dataobj, ct.dataobj)
    assert np.unique(merged.dataobj).tolist() == [2]


def test_source_change_after_discovery_is_rejected(tmp_path):
    m = module()
    image = tmp_path/'FUMPE/images/PAT001.nii.gz'; image.parent.mkdir(parents=True)
    label = tmp_path/'FUMPE/masks/PAT001.nii.gz'; label.parent.mkdir(parents=True)
    volume = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.float32),np.eye(4))
    nib.save(volume,image); nib.save(volume,label)
    record = m.discover(tmp_path)['fumpe'][0]
    with image.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='signature changed'):
        m.load_pair(record)


def test_qform_sform_disagreement_is_explicit_in_audit():
    m = module()
    image = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.float32),np.eye(4))
    label = nib.Nifti1Image(np.zeros((2,3,4),dtype=np.uint8),np.eye(4))
    qform = np.eye(4); qform[0,3] = 10
    image.set_qform(qform,code=1)
    _, _, audit = m._canonical_pair(image,label,'fumpe')
    assert audit['original_image_header']['qform_sform_disagreement']
    assert not audit['original_label_header']['qform_sform_disagreement']


def test_real_dicom_file_discovery_selects_referenced_series(tmp_path):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage, SegmentationStorage
    m = module()
    ids = {name:generate_uid() for name in ['ct','other','frame','ct0','ct1','ct2','seg']}

    def convert(item):
        ds = Dataset()
        for key, value in vars(item).items():
            if key == 'pixel_array': continue
            if key.endswith('UID'): value = ids[value]
            if isinstance(value,list) and value and isinstance(value[0],NS):
                value = [convert(x) for x in value]
            setattr(ds,key,value)
        return ds

    def save(ds,pixels,path,storage):
        ds.SOPClassUID = storage
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = pixels.dtype.itemsize*8; ds.BitsStored = ds.BitsAllocated
        ds.HighBit = ds.BitsStored-1; ds.PixelRepresentation = int(pixels.dtype.kind=='i')
        ds.PixelData = pixels.tobytes()
        ds.file_meta = FileMetaDataset(); ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = storage; ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        path.parent.mkdir(parents=True,exist_ok=True)
        pydicom.dcmwrite(path,ds,enforce_file_format=True)

    root = tmp_path/'Adrenal-ACC-Ki67-Seg'
    for z in range(3):
        native = ct_slice(z); ds = convert(native); ds.Modality = 'CT'
        save(ds,native.pixel_array,root/f'images/patient/study/ct/{z}.dcm',CTImageStorage)
    other = convert(ct_slice(0,series='other')); other.SOPInstanceUID = generate_uid(); other.Modality = 'CT'
    save(other,np.zeros((2,3),dtype=np.int16),root/'images/patient/study/other/0.dcm',CTImageStorage)
    native = seg(); ds = convert(native); ds.SOPInstanceUID = ids['seg']; ds.Modality = 'SEG'
    ds.Rows = 2; ds.Columns = 3; ds.NumberOfFrames = 2
    save(ds,native.pixel_array,root/'segmentations/patient/study/seg/seg.dcm',SegmentationStorage)
    record = m.discover(tmp_path)['adrenal'][0]
    assert record['case_id'] == 'patient' and len(record['ct_paths']) == 3
    image,label,audit = m.load_pair(record)
    assert image.shape == label.shape == (3,2,3)
    assert audit['case_id'] == 'patient' and audit['source_name'] == 'adrenal'
    assert len(audit['source_signatures']) == 4
