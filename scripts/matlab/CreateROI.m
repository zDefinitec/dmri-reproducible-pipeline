function CreateROI(dwifile, maskfile, outputfile)
%CREATEROI Stream a 4-D NIfTI into NODDI's masked row representation.

arguments
    dwifile (1,:) char
    maskfile (1,:) char
    outputfile (1,:) char
end

if exist('nifti', 'class') ~= 8 && exist('nifti', 'file') ~= 2
    error('dmri:noddi:NiftiMissing', ...
        'MATLAB NIfTI source is not available on the path.');
end
if ~isfile(dwifile) || ~isfile(maskfile)
    error('dmri:noddi:InputMissing', 'DWI and mask files must exist.');
end
if isfile(outputfile) || isfolder(outputfile)
    error('dmri:noddi:OutputExists', 'Refusing to overwrite ROI output.');
end

dwiObject = nifti(dwifile);
maskObject = nifti(maskfile);
dwiSize = double(dwiObject.dat.dim);
maskSize = double(maskObject.dat.dim);
if numel(dwiSize) < 4 || dwiSize(4) < 1
    error('dmri:noddi:DWIShape', 'DWI must be four-dimensional.');
end
if ~isequal(dwiSize(1:3), maskSize(1:3))
    error('dmri:noddi:GridMismatch', 'DWI and mask dimensions differ.');
end

maskValues = double(maskObject.dat(:, :, :));
if any(~isfinite(maskValues), 'all')
    error('dmri:noddi:MaskFinite', 'Mask contains non-finite values.');
end
linearIndices = find(maskValues > 0);
voxelCount = numel(linearIndices);
if voxelCount < 1
    error('dmri:noddi:MaskEmpty', 'Mask contains no ROI voxels.');
end
[ii, jj, kk] = ind2sub(dwiSize(1:3), linearIndices);
idx = [ii, jj, kk];
roi = zeros(voxelCount, dwiSize(4));

% Keep peak memory bounded to one scan volume plus the final ROI matrix.
for volumeIndex = 1:dwiSize(4)
    oneVolume = double(dwiObject.dat(:, :, :, volumeIndex));
    if any(~isfinite(oneVolume), 'all')
        error('dmri:noddi:DWIFinite', ...
            'DWI volume %d contains non-finite values.', volumeIndex);
    end
    roi(:, volumeIndex) = oneVolume(linearIndices);
end

mask = zeros(dwiSize(1:3), 'uint32');
mask(linearIndices) = uint32(1:voxelCount);
temporaryFile = [outputfile '.tmp'];
cleanup = onCleanup(@() deleteIfPresent(temporaryFile));
save(temporaryFile, 'roi', 'mask', 'idx', '-v7.3');
movefile(temporaryFile, outputfile);
clear cleanup
end


function deleteIfPresent(path)
if isfile(path)
    delete(path);
end
end
