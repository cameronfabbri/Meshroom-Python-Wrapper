"""

A Python wrapper around Meshroom executables

"""

import os
import cv2
import numpy as np

import sys
import uuid
import time
import argparse

import scripts.clean_mesh as clean_mesh

opj = os.path.join


def crop_images(image_paths):

    for image_path in image_paths:
        img = cv2.imread(image_path)
        edges = cv2.Canny(img, 10, 300)

        for i, row in enumerate(edges):
            if np.sum(row) > 0:
                start_y = i
                break
        for i, col in enumerate(edges.T):
            if np.sum(col) > 0:
                start_x = i
                break
        for i, row in enumerate(np.flipud(edges)):
            if np.sum(row) > 0:
                end_y = i
                break
        for i, col in enumerate(np.flipud(edges.T)):
            if np.sum(col) > 0:
                end_x = i
                break

        new_path = image_path.replace('images', 'cropped_images')
        cv2.imwrite(
            new_path, img[start_y-10:-(end_y-10), start_x-10:-(end_x-10), :])


class Meshroom:
    def __init__(
            self,
            meshroom_dir,
            project_dir,
            sensor_db=None,
            input_uuid=None):

        if input_uuid is None:
            self.uuid = str(uuid.uuid4())
        else:
            self.uuid = input_uuid

        self.meshroom_dir = meshroom_dir
        self.sift_tree_path = opj(
            meshroom_dir, 'aliceVision', 'share',
            'aliceVision', 'vlfeat_K80L3.SIFT.tree')

        self.image_dir = opj(project_dir, 'images')
        self.crop_dir = opj(project_dir, 'cropped_images')
        self.cache_dir = opj(project_dir, 'MeshroomCache')
        self.camera_dir = opj(self.cache_dir, 'CameraInit', self.uuid)
        self.feature_extraction_dir = opj(
            self.cache_dir, 'FeatureExtraction', self.uuid)
        self.feature_matching_dir = opj(
            self.cache_dir, 'FeatureMatching', self.uuid)
        self.image_matching_dir = opj(
            self.cache_dir, 'ImageMatching', self.uuid)
        self.sfm_dir = opj(self.cache_dir, 'StructureFromMotion', self.uuid)
        self.prepare_dense_dir = opj(
            self.cache_dir, 'PrepareDenseScene', self.uuid)
        self.depth_map_dir = opj(self.cache_dir, 'DepthMap', self.uuid)
        self.depth_map_filter_dir = opj(
            self.cache_dir, 'DepthMapFilter', self.uuid)
        self.meshing_dir = opj(
            self.cache_dir, 'Meshing', self.uuid)
        self.texturing_dir = opj(
            self.cache_dir, 'Texturing', self.uuid)
        self.log_dir = opj(self.cache_dir, 'logs', self.uuid)

        self.camera_sfm_file = opj(self.camera_dir, 'cameraInit.sfm')
        self.image_matching_file = opj(
            self.image_matching_dir, 'imageMatching.txt')
        self.sfm_file = opj(self.sfm_dir, 'sfm.abc')
        self.mesh_file = opj(
            self.meshing_dir, 'mesh.obj')
        self.mesh_file_filtered = opj(
            self.meshing_dir, 'mesh_filtered.obj')
        self.dense_point_cloud_file = opj(
            self.meshing_dir, 'densePointCloud.abc')

        #os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.sfm_dir, exist_ok=True)
        os.makedirs(self.camera_dir, exist_ok=True)
        os.makedirs(self.prepare_dense_dir, exist_ok=True)
        os.makedirs(self.image_matching_dir, exist_ok=True)
        os.makedirs(self.feature_extraction_dir, exist_ok=True)
        os.makedirs(self.feature_matching_dir, exist_ok=True)
        os.makedirs(self.depth_map_dir, exist_ok=True)
        os.makedirs(self.depth_map_filter_dir, exist_ok=True)
        os.makedirs(self.meshing_dir, exist_ok=True)
        os.makedirs(self.texturing_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if sensor_db is None:
            self.sensor_db = opj(
                self.meshroom_dir,
                'aliceVision', 'share', 'aliceVision', 'cameraSensors.db')
        else:
            self.sensor_db = sensor_db

        self.bin = opj(meshroom_dir, 'aliceVision', 'bin')

        #image_paths = [
        #    opj(self.image_dir, x) for x in os.listdir(self.image_dir)]
        #crop_images(image_paths)

    def camera_init(
            self,
            fov=45.,
            verbose_level='info',
            camera_models='pinhole,radial1,radial3,brown,fisheye4,fisheye1',
            view_id_method='metadata',
            allow_single_view=True,
            use_internal_white_balance=True):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_cameraInit.exe'))
        command.append('--imageFolder ' + self.image_dir)
        command.append('-s ' + self.sensor_db)
        command.append(' --defaultFieldOfView ' + str(fov))
        command.append('--groupCameraFallback folder')
        command.append('--allowedCameraModels ' + camera_models)
        command.append('--useInternalWhiteBalance True')
        command.append('--viewIdMethod metadata')
        command.append('--verboseLevel info')
        command.append('--output ' + self.camera_sfm_file)
        command.append('--allowSingleView 1')
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def feature_extraction(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_featureExtraction.exe'))
        command.append(' --input ' + self.camera_sfm_file)
        command.append('--describerTypes sift')
        command.append('--describerPreset normal')
        command.append('--describerQuality normal')
        command.append('--contrastFiltering GridSort')
        command.append('--gridFiltering True')
        command.append('--forceCpuExtraction False')
        command.append('--maxThreads 32')
        command.append('--verboseLevel info')
        command.append('--output ' + self.feature_extraction_dir)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def image_matching(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_imageMatching.exe'))
        command.append('--input ' + self.camera_sfm_file)
        command.append('--featuresFolders ' + self.feature_extraction_dir)
        command.append('--tree ' + self.sift_tree_path)
        command.append('--method VocabularyTree')
        command.append('--weights ""')
        command.append('--minNbImages 200')
        command.append('--maxDescriptors 500')
        command.append('--nbMatches 50')
        command.append('--verboseLevel info')
        command.append('--output ' + self.image_matching_file)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def feature_matching(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_featureMatching.exe'))
        command.append('--input ' + self.camera_sfm_file)
        command.append('--featuresFolders ' + self.feature_extraction_dir)
        command.append('--imagePairsList ' + self.image_matching_file)
        command.append('--describerTypes sift')
        command.append('--photometricMatchingMethod ANN_L2')
        command.append('--geometricEstimator acransac')
        command.append('--geometricFilterType fundamental_matrix')
        command.append('--distanceRatio 0.8')
        command.append('--maxIteration 2048')
        command.append('--geometricError 0.0')
        command.append('--knownPosesGeometricErrorMax 5.0')
        command.append('--maxMatches 0')
        command.append('--savePutativeMatches False')
        command.append('--crossMatching False')
        command.append('--guidedMatching True')
        command.append('--matchFromKnownCameraPoses False')
        command.append('--exportDebugFiles False')
        command.append('--verboseLevel info')
        command.append('--output ' + self.feature_matching_dir)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def structure_from_motion(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_incrementalSfM.exe'))
        command.append('--input ' + self.camera_sfm_file)
        command.append('--matchesFolder ' + self.feature_matching_dir)
        command.append('--featuresFolder ' + self.feature_extraction_dir)
        command.append('--describerTypes sift')
        command.append('--localizerEstimator acransac')
        command.append('--observationConstraint Basic')
        command.append('--localizerEstimatorMaxIterations 4096')
        command.append('--localizerEstimatorError 0.0')
        command.append('--lockScenePreviouslyReconstructed False')
        command.append('--useLocalBA True')
        command.append('--localBAGraphDistance 1')
        command.append('--maxNumberOfMatches 0')
        command.append('--minNumberOfMatches 0')
        command.append('--minInputTrackLength 2')
        command.append('--minNumberOfObservationsForTriangulation 2')
        command.append('--minAngleForTriangulation 3.0')
        command.append('--minAngleForLandmark 2.0')
        command.append('--maxReprojectionError 4.0')
        command.append('--minAngleInitialPair 5.0')
        command.append('--maxAngleInitialPair 40.0')
        command.append('--useOnlyMatchesFromInputFolder False')
        command.append('--useRigConstraint True')
        command.append('--lockAllIntrinsics False')
        command.append('--filterTrackForks False')
        command.append('--initialPairA ""')
        command.append('--initialPairB ""')
        command.append('--interFileExtension .abc')
        command.append('--output ' + self.sfm_file)
        command.append(
            '--outputViewsAndPoses ' + opj(self.sfm_dir, 'cameras.sfm'))
        command.append('--extraInfoFolder ' + self.sfm_dir)
        command.append('--verboseLevel info')
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def prepare_dense(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_prepareDenseScene.exe'))
        command.append('--input ' + self.sfm_file)
        command.append('--saveMetadata True')
        command.append('--saveMatricesTxtFiles False')
        command.append('--evCorrection False')
        command.append('--verboseLevel info')
        command.append('--output ' + self.prepare_dense_dir)
        command.append('--outputFileType exr')
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def compute_depth_map(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_depthMapEstimation.exe'))
        command.append('--input ' + self.sfm_file)
        command.append('--imagesFolder ' + self.prepare_dense_dir)
        command.append('--output ' + self.depth_map_dir)

        command.append('--downscale 2')
        command.append('--minViewAngle 2.0')
        command.append('--maxViewAngle 70.0')
        command.append('--sgmMaxTCams 10')
        command.append('--sgmWSH 4')
        command.append('--sgmGammaC 5.5')
        command.append('--sgmGammaP 8.0')
        command.append('--refineMaxTCams 6')
        command.append('--refineNSamplesHalf 150')
        command.append('--refineNDepthsToRefine 31')
        command.append('--refineNiters 100')
        command.append('--refineWSH 3')
        command.append('--refineSigma 15')
        command.append('--refineGammaC 15.5')
        command.append('--refineGammaP 8.0')
        command.append('--refineUseTcOrRcPixSize False')
        command.append('--exportIntermediateResults False')
        command.append('--nbGPUs 0')
        command.append('--verboseLevel info')
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def compute_depth_map_filter(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_depthMapFiltering.exe'))
        command.append('--input ' + self.sfm_file)
        command.append('--depthMapsFolder ' + self.depth_map_dir)
        command.append('--minViewAngle 2.0')
        command.append('--maxViewAngle 70.0')
        command.append('--nNearestCams 10')
        command.append('--minNumOfConsistentCams 3')
        command.append('--minNumOfConsistentCamsWithLowSimilarity 4')
        command.append('--pixSizeBall 0')
        command.append('--pixSizeBallWithLowSimilarity 0')
        command.append('--computeNormalMaps False')
        command.append('--verboseLevel info')
        command.append('--output ' + self.depth_map_filter_dir)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def compute_mesh(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_meshing.exe'))
        command.append('--input ' + self.sfm_file)
        command.append('--depthMapsFolder ' + self.depth_map_filter_dir)
        command.append('--estimateSpaceFromSfM True')
        command.append('--estimateSpaceMinObservations 3')
        command.append('--estimateSpaceMinObservationAngle 10')
        command.append('--maxInputPoints 50000000')
        command.append('--maxPoints 5000000')
        command.append('--maxPointsPerVoxel 1000000')
        command.append('--minStep 2')
        command.append('--partitioning singleBlock')
        command.append('--repartition multiResolution')
        command.append('--angleFactor 15.0')
        command.append('--simFactor 15.0')
        command.append('--pixSizeMarginInitCoef 2.0')
        command.append('--pixSizeMarginFinalCoef 4.0')
        command.append('--voteMarginFactor 4.0')
        command.append('--contributeMarginFactor 2.0')
        command.append('--simGaussianSizeInit 10.0')
        command.append('--simGaussianSize 10.0')
        command.append('--minAngleThreshold 1.0')
        command.append('--refineFuse True')
        command.append('--helperPointsGridSize 10')
        command.append('--nPixelSizeBehind 4.0')
        command.append('--fullWeight 1.0')
        command.append('--voteFilteringForWeaklySupportedSurfaces True')
        command.append('--addLandmarksToTheDensePointCloud False')
        command.append('--invertTetrahedronBasedOnNeighborsNbIterations 10')
        command.append('--minSolidAngleRatio 0.2')
        command.append('--nbSolidAngleFilteringIterations 2')
        command.append('--colorizeOutput False')
        command.append('--maxNbConnectedHelperPoints 50')
        command.append('--saveRawDensePointCloud False')
        command.append('--exportDebugTetrahedralization False')
        command.append('--seed 0')
        command.append('--verboseLevel info')
        command.append('--outputMesh ' + self.mesh_file)
        command.append('--output ' + self.dense_point_cloud_file)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def filter_mesh(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_meshFiltering.exe'))
        command.append('--input ' + self.mesh_file)
        command.append('--keepLargestMeshOnly True')
        command.append('--smoothingSubset all')
        command.append('--smoothingBoundariesNeighbours 0')
        command.append('--smoothingIterations 5')
        command.append('--smoothingLambda 1.0')
        command.append('--filteringSubset all')
        command.append('--filteringIterations 1')
        command.append('--filterLargeTrianglesFactor 60.0')
        command.append('--filterTrianglesRatio 0.0')
        command.append('--verboseLevel info')
        command.append('--outputMesh ' + self.mesh_file_filtered)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

    def compute_texture(self):
        s = time.time()
        command = []
        command.append(opj(self.bin, 'aliceVision_texturing.exe'))
        command.append('--input ' + self.dense_point_cloud_file)
        command.append('--imagesFolder ' + self.prepare_dense_dir)
        command.append('--inputMesh ' + self.mesh_file_filtered)
        command.append('--textureSide 8192')
        command.append('--downscale 2')
        command.append('--outputTextureFileType png')
        command.append('--unwrapMethod Basic')
        command.append('--useUDIM True')
        command.append('--fillHoles False')
        command.append('--padding 5')
        command.append('--multiBandDownscale 4')
        command.append('--multiBandNbContrib 1 5 10 0')
        command.append('--useScore True')
        command.append('--bestScoreThreshold 0.1')
        command.append('--angleHardThreshold 90.0')
        command.append('--processColorspace sRGB')
        command.append('--correctEV False')
        command.append('--forceVisibleByAllVertices False')
        command.append('--flipNormals False')
        command.append('--visibilityRemappingMethod PullPush')
        command.append('--subdivisionTargetRatio 0.8')
        command.append('--verboseLevel info')
        command.append('--output ' + self.texturing_dir)
        command = ' '.join(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(command)
        os.system(command)
        with open(opj(self.log_dir, self.uuid+'.log'), 'a') as f:
            f.write(' - time: ' + str(time.time() - s) + '\n')

        # Run cleaning script
        print('Cleaning final mesh...')
        clean_mesh.main(opj(self.texturing_dir, 'texturedMesh.obj'))
        print('Done')


def main(argv):

    args = parse(argv)

    s = time.time()

    for i, project_dir in enumerate(args.project_dirs.split(',')):
        if args.multicore:
            # Start a screen session
            pc = 'python -m scripts.run_meshroom --root_dir='+args.root_dir+' --project_dirs='+project_dir
            c = 'screen -S sess' + str(i) + ' -d -m bash -c "' + pc + '"'
            os.system(c)
        else:
            meshroom = Meshroom(
                meshroom_dir=args.meshroom_dir,
                project_dir=opj(args.root_dir, project_dir))

            meshroom.camera_init()
            meshroom.feature_extraction()
            meshroom.image_matching()
            meshroom.feature_matching()
            meshroom.structure_from_motion()
            meshroom.prepare_dense()
            meshroom.compute_depth_map()
            meshroom.compute_depth_map_filter()
            meshroom.compute_mesh()
            meshroom.filter_mesh()
            meshroom.compute_texture()

    c = time.time() - s
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    print('\n')
    print('Total Time')
    print('----------')
    print(days, 'days')
    print(hours, 'hours')
    print(minutes, 'minutes')
    print(seconds, 'seconds')


def parse(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--root_dir', type=str, required=False, default='data/photogrammetry',
        help='Root directory where project directories are stored')
    parser.add_argument(
        '--project_dirs', type=str, required=True,
        help='Directories separated by a comma where the images/ folder is located')
    parser.add_argument(
        '--meshroom_dir', type=str, required=False, default='meshroom',
        help='Root directory for meshroom')
    parser.add_argument(
        '--multicore', action='store_true',
        help='If enabled, will spawn one python script per project dir')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    main(sys.argv)
