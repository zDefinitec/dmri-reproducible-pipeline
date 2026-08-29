function merge_noddi_workers(root, numWorkers)
%MERGE_NODDI_WORKERS Validate and merge exact contiguous worker blocks.

root = validateRoot(root);
validateattributes(numWorkers, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});
scriptDirectory = fileparts(mfilename('fullpath'));
packageRoot = fileparts(fileparts(scriptDirectory));
toolboxRoot = fullfile(packageRoot, 'vendor', 'noddi_toolbox_v1.05');
compatRoot = fullfile(root, 'nifti_matlab', 'matlab');
addpath(compatRoot, '-begin');
addpath(fullfile(toolboxRoot, 'fitting'));
addpath(genpath(fullfile(toolboxRoot, 'models')));

roiPath = fullfile(root, 'NODDI_roi.mat');
roiStore = matfile(roiPath);
roiSize = size(roiStore, 'roi');
totalRows = roiSize(1);
numMeasurements = roiSize(2);
model = MakeModel('WatsonSHStickTortIsoV_B0');
numParams = model.numParams;
blockSize = ceil(totalRows / numWorkers);
preparePath = fullfile(root, 'noddi_prepare.json');
bvalsPath = fullfile(root, 'bvals_rounded.txt');
bvecsPath = fullfile(root, 'eddy_rotated_bvecs.txt');
workerSourcePath = fullfile(scriptDirectory, 'run_noddi_worker.m');
requiredCurrent = struct( ...
    'roiHash', sha256File(roiPath), ...
    'bvalsHash', sha256File(bvalsPath), ...
    'bvecsHash', sha256File(bvecsPath), ...
    'prepareManifestHash', sha256File(preparePath), ...
    'workerSourceHash', sha256File(workerSourcePath), ...
    'noddiSourceHash', sha256Tree(toolboxRoot), ...
    'matlabVersion', version, ...
    'mexExtension', mexext);
mexNames = {'file2mat', 'mat2file', 'init'};
mexHashes = cell(1, numel(mexNames));
for mexIndex = 1:numel(mexNames)
    mexHashes{mexIndex} = sha256File(fullfile( ...
        compatRoot, '@file_array', 'private', ...
        [mexNames{mexIndex} '.' mexext]));
end
requiredCurrent.mexHashes = mexHashes;

gsps = nan(totalRows, numParams);
mlps = nan(totalRows, numParams);
fobj_gs = nan(totalRows, 1);
fobj_ml = nan(totalRows, 1);
error_code = nan(totalRows, 1);
first999Exceptions = emptyExceptionSamples();
covered = false(totalRows, 1);

for workerIndex = 1:numWorkers
    expectedStart = (workerIndex - 1) * blockSize + 1;
    expectedEnd = min(workerIndex * blockSize, totalRows);
    if expectedStart > expectedEnd
        error('dmri:noddi:EmptyWorker', ...
            'Worker count exceeds the number of ROI rows.');
    end
    path = fullfile(root, sprintf('worker_%02d_final.mat', workerIndex));
    if ~isfile(path)
        error('dmri:noddi:WorkerMissing', ...
            'Missing worker final: %s', path);
    end
    worker = load(path);
    required = {'gsps', 'mlps', 'fobj_gs', 'fobj_ml', ...
        'error_code', 'first999Exceptions', 'nextRow', 'metadata'};
    if ~all(isfield(worker, required))
        error('dmri:noddi:WorkerFields', ...
            'Worker %d final is structurally incomplete.', workerIndex);
    end
    metadata = worker.metadata;
    expectedFields = {'schemaVersion', 'workerIndexStored', 'numWorkersStored', ...
        'globalStart', 'globalEnd', 'totalRows', 'numMeasurements', ...
        'numParams', 'modelName'};
    if ~all(isfield(metadata, expectedFields)) || ...
            metadata.schemaVersion ~= 1 || ...
            metadata.workerIndexStored ~= workerIndex || ...
            metadata.numWorkersStored ~= numWorkers || ...
            metadata.globalStart ~= expectedStart || ...
            metadata.globalEnd ~= expectedEnd || ...
            metadata.totalRows ~= totalRows || ...
            metadata.numMeasurements ~= numMeasurements || ...
            metadata.numParams ~= numParams || ...
            ~strcmp(metadata.modelName, model.name)
        error('dmri:noddi:WorkerMetadata', ...
            'Worker %d metadata is inconsistent.', workerIndex);
    end
    currentFields = fieldnames(requiredCurrent);
    for fieldIndex = 1:numel(currentFields)
        fieldName = currentFields{fieldIndex};
        if ~isfield(metadata, fieldName) || ...
                ~isequal(metadata.(fieldName), requiredCurrent.(fieldName))
            error('dmri:noddi:WorkerSourceMetadata', ...
                'Worker %d source metadata mismatch: %s.', ...
                workerIndex, fieldName);
        end
    end
    localRows = expectedEnd - expectedStart + 1;
    if worker.nextRow ~= localRows + 1 || ...
            ~isequal(size(worker.gsps), [localRows, numParams]) || ...
            ~isequal(size(worker.mlps), [localRows, numParams]) || ...
            ~isequal(size(worker.fobj_gs), [localRows, 1]) || ...
            ~isequal(size(worker.fobj_ml), [localRows, 1]) || ...
            ~isequal(size(worker.error_code), [localRows, 1])
        error('dmri:noddi:WorkerShape', ...
            'Worker %d final has invalid row or parameter dimensions.', ...
            workerIndex);
    end
    validateExceptionSamples( ...
        worker.first999Exceptions, worker.error_code, ...
        workerIndex, expectedStart, 3);
    first999Exceptions = [first999Exceptions, ...
        reshape(worker.first999Exceptions, 1, [])]; %#ok<AGROW>
    rows = expectedStart:expectedEnd;
    if any(covered(rows))
        error('dmri:noddi:WorkerOverlap', ...
            'Worker %d overlaps an earlier block.', workerIndex);
    end
    covered(rows) = true;
    gsps(rows, :) = worker.gsps;
    mlps(rows, :) = worker.mlps;
    fobj_gs(rows) = worker.fobj_gs;
    fobj_ml(rows) = worker.fobj_ml;
    error_code(rows) = worker.error_code;
end
if ~all(covered)
    error('dmri:noddi:WorkerGap', 'Worker blocks contain a gap.');
end

paramsPath = fullfile(root, 'NODDI_params.mat');
temporaryParams = [paramsPath '.tmp'];
save(temporaryParams, 'model', 'gsps', 'mlps', ...
    'fobj_gs', 'fobj_ml', 'error_code', '-v7.3');
movefile(temporaryParams, paramsPath, 'f');

targetPath = fullfile(root, 'cleaned_mask.nii');
outputPrefix = fullfile(root, 'NODDI');
SaveParamsAsNIfTI(paramsPath, roiPath, targetPath, outputPrefix);

successCount = nnz(error_code == 0);
error999Count = nnz(error_code == 999);
otherErrorCount = totalRows - successCount - error999Count;
finiteObjectives = fobj_ml(isfinite(fobj_ml));
if isempty(finiteObjectives)
    objectiveMinimum = nan;
    objectiveMaximum = nan;
    objectiveMean = nan;
else
    objectiveMinimum = min(finiteObjectives);
    objectiveMaximum = max(finiteObjectives);
    objectiveMean = mean(finiteObjectives);
end
listing = dir(fullfile(root, 'NODDI_*.nii'));
parameterMaps = sort({listing.name});
exceptionMetrics = reshape(num2cell(first999Exceptions), 1, []);
metrics = struct( ...
    'total_voxels', totalRows, ...
    'success_count', successCount, ...
    'error_999_count', error999Count, ...
    'other_error_count', otherErrorCount, ...
    'objective_finite_count', numel(finiteObjectives), ...
    'objective_min', objectiveMinimum, ...
    'objective_max', objectiveMaximum, ...
    'objective_mean', objectiveMean, ...
    'worker_count', numWorkers, ...
    'model_name', model.name, ...
    'parameter_maps', {parameterMaps}, ...
    'first_999_exceptions', {exceptionMetrics});
jsonPath = fullfile(root, 'noddi_metrics.json');
temporaryJson = [jsonPath '.tmp'];
descriptor = fopen(temporaryJson, 'w');
if descriptor < 0
    error('dmri:noddi:MetricsOpen', ...
        'Cannot open metrics JSON for writing.');
end
try
    fprintf(descriptor, '%s\n', ...
        jsonencode(metrics, 'PrettyPrint', true));
catch writeException
    fclose(descriptor);
    rethrow(writeException);
end
fclose(descriptor);
movefile(temporaryJson, jsonPath, 'f');
end


function samples = emptyExceptionSamples()
samples = struct( ...
    'worker', {}, ...
    'global_row', {}, ...
    'identifier', {}, ...
    'message', {}, ...
    'report', {});
end


function validateExceptionSamples(samples, errors, workerIndex, firstRow, maximum)
required = {'worker', 'global_row', 'identifier', 'message', 'report'};
if ~isstruct(samples) || numel(samples) > maximum || ...
        ~all(isfield(samples, required))
    error('dmri:noddi:WorkerExceptions', ...
        'Worker %d has invalid 999 exception samples.', workerIndex);
end
failedLocalRows = find(errors == 999, maximum, 'first');
expectedGlobalRows = firstRow + failedLocalRows - 1;
if numel(samples) ~= numel(expectedGlobalRows)
    error('dmri:noddi:WorkerExceptions', ...
        'Worker %d has invalid 999 exception samples.', workerIndex);
end
for index = 1:numel(samples)
    sample = samples(index);
    if ~isnumeric(sample.worker) || ~isscalar(sample.worker) || ...
            ~isfinite(sample.worker) || fix(sample.worker) ~= sample.worker || ...
            ~isnumeric(sample.global_row) || ~isscalar(sample.global_row) || ...
            ~isfinite(sample.global_row) || ...
            fix(sample.global_row) ~= sample.global_row || ...
            sample.worker ~= workerIndex || ...
            sample.global_row ~= expectedGlobalRows(index) || ...
            ~ischar(sample.identifier) || ...
            ~ischar(sample.message) || ~ischar(sample.report)
        error('dmri:noddi:WorkerExceptions', ...
            'Worker %d has invalid 999 exception samples.', workerIndex);
    end
end
end


function root = validateRoot(root)
if isstring(root) && isscalar(root)
    root = char(root);
end
if ~ischar(root) || isempty(root) || ~isfolder(root) || ...
        ~startsWith(root, filesep) || contains(root, [filesep '..' filesep])
    error('dmri:noddi:Root', ...
        'root must name an absolute existing traversal-free directory.');
end
canonical = char(java.io.File(root).getCanonicalPath());
if ~strcmp(canonical, root)
    error('dmri:noddi:Root', ...
        'root must be canonical and must not traverse symbolic links.');
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
