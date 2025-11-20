# MRE_habitat
HCC Intratumoral Biomechanical Heterogeneity by Habitat Analysis

# program usage in cmd:
__(1) Super-voxel segmentation by SLIC algorithm：__
<br>
cMap:
python D:\projects\MRE\programs\mre_slic_supervoxels.py   --root_dir "D:\projects\MRE\data"  --enable_2d_auto --max_slices_for_2d 3  --compactness_2d 0.15 --min_size_vox_2d 3 --tiny_roi_vox 100 --tiny_policy single --target_size_mm3 700 --max_segments 100 --min_size_vox 3 --z_boost 2  --channels c --export_features --features_used_only --out_name supervoxels_slic_c.nii.gz --features_out_name supervoxels_features_c.csv

phiMap:
python D:\projects\MRE\programs\mre_slic_supervoxels.py   --root_dir "D:\projects\MRE\data"  --enable_2d_auto --max_slices_for_2d 3  --compactness_2d 0.15 --min_size_vox_2d 3 --tiny_roi_vox 100 --tiny_policy single  --target_size_mm3 700 --max_segments 100 --min_size_vox 3 --z_boost 2 --channels phi --export_features --features_used_only --out_name supervoxels_slic_phi.nii.gz --features_out_name supervoxels_features_phi.csv


__（2）Habitat generation by k-means clustering：__
<br>
cMap:
python D:\projects\MRE\programs\mre_cohort_habitat_clustering.py --root_dir "D:\projects\MRE\data" --mode both --k_range 3:6 --scaler robust --min_size_vox_train 5 --features_csv_name supervoxels_features_c.csv --labels_nii_name supervoxels_slic_c.nii.gz --out_habitat_name habitat_map_c.nii.gz --out_assign_name supervoxels_habitat_c.csv --model_dir "D:\projects\MRE\habitat_model_c"

phiMap:
python D:\projects\MRE\programs\mre_cohort_habitat_clustering.py --root_dir "D:\projects\data" --mode both --k_range 3:6 --scaler robust --min_size_vox_train 5 --features_csv_name supervoxels_features_phi.csv --labels_nii_name supervoxels_slic_phi.nii.gz --out_habitat_name habitat_map_phi.nii.gz --out_assign_name supervoxels_habitat_phi.csv --model_dir "D:\projects\MRE\habitat_model_phi"


__（3）Habitats feature calculation：__
<br>
cMap:
python D:\projects\MRE\programs\make_habitat_patient_table-means.py --root_dir "D:\projects\MRE\data" --features_csv_name supervoxels_features_c.csv --habitat_csv_name supervoxels_habitat_c.csv --k 3 --out_excel "patient_habitat_features_k3_c_means.xlsx" --out_dir "D:\projects\MRE\"
 
phiMap:
python D:\projects\MRE\programs\make_habitat_patient_table-means.py --root_dir "D:\projects\MRE\data" --features_csv_name supervoxels_features_phi.csv --habitat_csv_name supervoxels_habitat_phi.csv --k 5 --out_excel "patient_habitat_features_k5_phi_means.xlsx" --out_dir "D:\projects\MRE\"
