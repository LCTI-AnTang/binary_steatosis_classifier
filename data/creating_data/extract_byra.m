data = 'I:\Chercheurs\Cloutier_Guy\Individuel\Pedro vianna\Code\dataset_liver_bmodes_steatosis_assessment_IJCARS.mat';
path = 'I:\Chercheurs\Cloutier_Guy\Individuel\Pedro vianna\Code\byra_dataset\';
S = load(data);
dataset = S.data;

cell = struct2cell(dataset);

for i = 1:length(cell)
    pat = cell(:,:,i);
    id = pat{1};
    class = pat{2};
    fat = pat{3};
    images = pat{4};

    for j = 1:10
        single_img = images(j,:,:);
        prefinal_img = squeeze(single_img);
        final_img = prefinal_img./255;
        name = sprintf('ID_%d_Class_%d_Fat_%d_Image_%d.jpeg',id, class,fat,j);
        fname = fullfile(path,name);
        imwrite(final_img,name)
    end
end

        