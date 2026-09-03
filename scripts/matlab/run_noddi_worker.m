function run_noddi_worker(root, workerIndex, numWorkers)
%RUN_NODDI_WORKER Fit one deterministic contiguous ROI block with checkpoints.

root = validateRoot(root);
validateattributes(workerIndex, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});
validateattributes(numWorkers, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});
if workerIndex > numWorkers
    error('dmri:noddi:WorkerIndex', ...
        'workerIndex must not exceed numWorkers.');
end

scriptPath = [mfilename('fullpath') '.m'];
if ~isfile(scriptPath)
    error('dmri:noddi:WorkerSourceMissing', ...
        'Worker source file is missing: %s', scriptPath);
end
scriptDirectory = fileparts(scriptPath);
packageRoot = fileparts(fileparts(scriptDirectory));
toolboxRoot = fullfile(packageRoot, 'vendor', 'noddi_toolbox_v1.05');
compatRoot = fullfile(root, 'nifti_matlab', 'matlab');
requireDirectory(fullfile(toolboxRoot, 'fitting'), 'NODDI fitting source');
requireDirectory(fullfile(toolboxRoot, 'models'), 'NODDI model source');
requireDirectory(compatRoot, 'stage-local NIfTI compatibility source');
addpath(scriptDirectory, '-begin');
addpath(compatRoot, '-begin');
addpath(fullfile(toolboxRoot, 'fitting'));
addpath(genpath(fullfile(toolboxRoot, 'models')));

roiPath = fullfile(root, 'NODDI_roi.mat');
bvalsPath = fullfile(root, 'bvals_rounded.txt');
bvecsPath = fullfile(root, 'eddy_rotated_bvecs.txt');
preparePath = fullfile(root, 'noddi_prepare.json');
requireFile(roiPath, 'ROI');
requireFile(bvalsPath, 'rounded b-values');
requireFile(bvecsPath, 'EDDY-rotated b-vectors');
requireFile(preparePath, 'preparation manifest');

roiStore = matfile(roiPath);
roiSize = size(roiStore, 'roi');
if numel(roiSize) ~= 2 || any(roiSize < 1)
    error('dmri:noddi:ROIShape', 'ROI must be a non-empty matrix.');
end
totalRows = roiSize(1);
numMeasurements = roiSize(2);
bvals = readmatrix(bvalsPath, 'FileType', 'text');
bvecs = readmatrix(bvecsPath, 'FileType', 'text');
bvals = reshape(bvals, 1, []);
if ~isequal(size(bvecs), [3, numMeasurements]) || ...
        numel(bvals) ~= numMeasurements || ...
        any(~isfinite(bvals), 'all') || any(~isfinite(bvecs), 'all')
    error('dmri:noddi:ProtocolShape', ...
        'Protocol arrays do not match the ROI measurements.');
end
protocol = FSL2Protocol(bvalsPath, bvecsPath);
model = MakeModel('WatsonSHStickTortIsoV_B0');
numParams = model.numParams;
mexDirectory = fullfile(compatRoot, '@file_array', 'private');
mexHashes = cell(1, 3);
mexNames = {'file2mat', 'mat2file', 'init'};
for mexIndex = 1:numel(mexNames)
    mexPath = fullfile(mexDirectory, ...
        [mexNames{mexIndex} '.' mexext]);
    requireFile(mexPath, 'compiled NIfTI MEX');
    mexHashes{mexIndex} = sha256File(mexPath);
end

blockSize = ceil(totalRows / numWorkers);
globalStart = (workerIndex - 1) * blockSize + 1;
globalEnd = min(workerIndex * blockSize, totalRows);
if globalStart > globalEnd
    error('dmri:noddi:EmptyBlock', ...
        'Worker count exceeds the number of ROI rows.');
end
localRows = globalEnd - globalStart + 1;

metadata = struct( ...
    'schemaVersion', 1, ...
    'workerIndexStored', workerIndex, ...
    'numWorkersStored', numWorkers, ...
    'globalStart', globalStart, ...
    'globalEnd', globalEnd, ...
    'totalRows', totalRows, ...
    'numMeasurements', numMeasurements, ...
    'numParams', numParams, ...
    'modelName', model.name, ...
    'roiHash', sha256File(roiPath), ...
    'bvalsHash', sha256File(bvalsPath), ...
    'bvecsHash', sha256File(bvecsPath), ...
    'prepareManifestHash', sha256File(preparePath), ...
    'workerSourceHash', sha256File(scriptPath), ...
    'noddiSourceHash', sha256Tree(toolboxRoot), ...
    'mexHashes', {mexHashes}, ...
    'matlabVersion', version, ...
    'mexExtension', mexext);

finalPath = fullfile(root, sprintf('worker_%02d_final.mat', workerIndex));
checkpointPath = fullfile(root, ...
    sprintf('worker_%02d_checkpoint.mat', workerIndex));
if isfile(finalPath)
    validateSavedFit(finalPath, metadata, localRows, numParams, true);
    fprintf('Worker %d final is valid; skipping.\n', workerIndex);
    return
end

gsps = nan(localRows, numParams);
mlps = nan(localRows, numParams);
fobj_gs = nan(localRows, 1);
fobj_ml = nan(localRows, 1);
error_code = nan(localRows, 1);
first999Exceptions = emptyExceptionSamples();
nextRow = 1;
if isfile(checkpointPath)
    saved = validateSavedFit( ...
        checkpointPath, metadata, localRows, numParams, false);
    gsps = saved.gsps;
    mlps = saved.mlps;
    fobj_gs = saved.fobj_gs;
    fobj_ml = saved.fobj_ml;
    error_code = saved.error_code;
    first999Exceptions = saved.first999Exceptions;
    nextRow = saved.nextRow;
end

checkpointEvery = 500;
for localRow = nextRow:localRows
    globalRow = globalStart + localRow - 1;
    signal = double(roiStore.roi(globalRow, :)).';
    try
        [gsps(localRow, :), fobj_gs(localRow), ...
            mlps(localRow, :), fobj_ml(localRow), ...
            error_code(localRow)] = ...
            ThreeStageFittingVoxel(signal, protocol, model, 0);
    catch voxelException
        if numel(first999Exceptions) < 3
            first999Exceptions(end + 1) = exceptionSample( ...
                workerIndex, globalRow, voxelException); %#ok<AGROW>
            warning('dmri:noddi:VoxelFit', ...
                'Worker %d voxel row %d failed: %s', ...
                workerIndex, globalRow, voxelException.message);
        end
        gsps(localRow, :) = nan;
        mlps(localRow, :) = nan;
        fobj_gs(localRow) = nan;
        fobj_ml(localRow) = nan;
        error_code(localRow) = 999;
    end
    nextRow = localRow + 1;
    if mod(localRow, checkpointEvery) == 0 || localRow == localRows
        atomicSave(checkpointPath, gsps, mlps, fobj_gs, fobj_ml, ...
            error_code, first999Exceptions, nextRow, metadata);
    end
end

atomicSave(finalPath, gsps, mlps, fobj_gs, fobj_ml, ...
    error_code, first999Exceptions, localRows + 1, metadata);
validateSavedFit(finalPath, metadata, localRows, numParams, true);
end


function root = validateRoot(root)
if isstring(root) && isscalar(root)
    root = char(root);
end
if ~ischar(root) || isempty(root) || ~isfolder(root)
    error('dmri:noddi:Root', 'root must name an existing directory.');
end
if ~startsWith(root, filesep) || contains(root, [filesep '..' filesep])
    error('dmri:noddi:Root', 'root must be absolute and traversal-free.');
end
canonical = char(java.io.File(root).getCanonicalPath());
if ~strcmp(canonical, root)
    error('dmri:noddi:Root', ...
        'root must be canonical and must not traverse symbolic links.');
end
end


function saved = validateSavedFit(path, expected, localRows, numParams, final)
saved = load(path);
required = {'gsps', 'mlps', 'fobj_gs', 'fobj_ml', 'error_code', ...
    'first999Exceptions', 'nextRow', 'metadata'};
if ~all(isfield(saved, required))
    error('dmri:noddi:ResumeFields', ...
        'Worker resume file lacks required checkpoint variables.');
end
metadataFields = fieldnames(expected);
for index = 1:numel(metadataFields)
    name = metadataFields{index};
    if ~isfield(saved.metadata, name) || ...
            ~isequal(saved.metadata.(name), expected.(name))
        error('dmri:noddi:ResumeMetadata', ...
            'Worker resume metadata mismatch: %s.', name);
    end
end
if ~isequal(size(saved.gsps), [localRows, numParams]) || ...
        ~isequal(size(saved.mlps), [localRows, numParams]) || ...
        ~isequal(size(saved.fobj_gs), [localRows, 1]) || ...
        ~isequal(size(saved.fobj_ml), [localRows, 1]) || ...
        ~isequal(size(saved.error_code), [localRows, 1])
    error('dmri:noddi:ResumeShape', ...
        'Worker resume arrays have inconsistent dimensions.');
end
validateattributes(saved.nextRow, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1, '<=', localRows + 1});
if final && saved.nextRow ~= localRows + 1
    error('dmri:noddi:FinalIncomplete', ...
        'Worker final file does not contain a complete block.');
end
validateExceptionSamples( ...
    saved.first999Exceptions, saved.error_code, expected, 3);
end


function atomicSave(path, gsps, mlps, fobj_gs, fobj_ml, ...
        error_code, first999Exceptions, nextRow, metadata)
temporaryPath = [path '.tmp'];
cleanup = onCleanup(@() deleteIfPresent(temporaryPath));
save(temporaryPath, 'gsps', 'mlps', 'fobj_gs', 'fobj_ml', ...
    'error_code', 'first999Exceptions', 'nextRow', 'metadata', '-v7.3');
movefile(temporaryPath, path, 'f');
clear cleanup
end


function samples = emptyExceptionSamples()
samples = struct( ...
    'worker', {}, ...
    'global_row', {}, ...
    'identifier', {}, ...
    'message', {}, ...
    'report', {});
end


function sample = exceptionSample(workerIndex, globalRow, caught)
try
    extendedReport = getReport(caught, 'extended', 'hyperlinks', 'off');
catch
    extendedReport = caught.message;
end
sample = struct( ...
    'worker', workerIndex, ...
    'global_row', globalRow, ...
    'identifier', caught.identifier, ...
    'message', caught.message, ...
    'report', extendedReport);
end


function validateExceptionSamples(samples, errors, metadata, maximum)
required = {'worker', 'global_row', 'identifier', 'message', 'report'};
if ~isstruct(samples) || numel(samples) > maximum || ...
        ~all(isfield(samples, required))
    error('dmri:noddi:ResumeExceptions', ...
        'Worker resume file contains invalid 999 exception samples.');
end
failedLocalRows = find(errors == 999, maximum, 'first');
expectedGlobalRows = metadata.globalStart + failedLocalRows - 1;
if numel(samples) ~= numel(expectedGlobalRows)
    error('dmri:noddi:ResumeExceptions', ...
        'Worker resume file contains invalid 999 exception samples.');
end
for index = 1:numel(samples)
    sample = samples(index);
    if ~isnumeric(sample.worker) || ~isscalar(sample.worker) || ...
            ~isfinite(sample.worker) || fix(sample.worker) ~= sample.worker || ...
            ~isnumeric(sample.global_row) || ~isscalar(sample.global_row) || ...
            ~isfinite(sample.global_row) || ...
            fix(sample.global_row) ~= sample.global_row || ...
            sample.worker ~= metadata.workerIndexStored || ...
            sample.global_row ~= expectedGlobalRows(index) || ...
            ~ischar(sample.identifier) || ~ischar(sample.message) || ...
            ~ischar(sample.report)
        error('dmri:noddi:ResumeExceptions', ...
            'Worker resume file contains invalid 999 exception samples.');
    end
end
end


function digest = sha256File(path)
engine = java.security.MessageDigest.getInstance('SHA-256');
stream = java.io.FileInputStream(java.io.File(path));
digestStream = java.security.DigestInputStream(stream, engine);
cleanup = onCleanup(@() digestStream.close());
buffer = zeros(1, 1024 * 1024, 'int8');
while true
    count = digestStream.read(buffer, 0, numel(buffer));
    if count < 0
        break
    end
end
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
clear cleanup
end


function digest = sha256Tree(root)
listing = dir(fullfile(root, '**', '*'));
listing = listing(~[listing.isdir]);
paths = sort(fullfile({listing.folder}, {listing.name}));
engine = java.security.MessageDigest.getInstance('SHA-256');
for index = 1:numel(paths)
    relative = erase(paths{index}, [root filesep]);
    engine.update(int8(unicode2native(relative, 'UTF-8')));
    engine.update(int8(0));
    engine.update(int8(unicode2native(sha256File(paths{index}), 'UTF-8')));
end
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end


function requireFile(path, label)
if ~isfile(path)
    error('dmri:noddi:MissingFile', '%s is missing: %s', label, path);
end
end


function requireDirectory(path, label)
if ~isfolder(path)
    error('dmri:noddi:MissingDirectory', '%s is missing: %s', label, path);
end
end


function deleteIfPresent(path)
if isfile(path)
    delete(path);
end
end
