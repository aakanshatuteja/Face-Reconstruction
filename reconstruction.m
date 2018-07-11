clc; clear all; close all;
A = load('faces.mat')
[m, n] = size(A.FACES);
i_size = sqrt(m); s = sqrt(n); grid_size = i_size * s;

%%%%%%%%%%% DISPLAYING THE DATABASE %%%%%%%%%%
for i = 1:n
   k = A.FACES(:,i);
   image{i} = reshape(k, [i_size, i_size]);
end
f = struct('faces', []);
p = 1;
for i = 1: i_size: grid_size
   for j = 1: i_size: grid_size
      f.faces(i: i + (i_size-1) , j: j + (i_size-1)) = image{p};
      p = p + 1;
   end
end
images = f.faces;
figure(1); imshow(images, []);title('IMAGE DATABASE')

%%%%%%%%%%%%%% MEAN FACE %%%%%%%%%%%%%%%%%%%%%%
B = A.FACES;
face = mean(B,2);
m_face = reshape(face, [i_size, i_size]);
figure(2); imshow(m_face, []);title('MEAN FACE')

%%%%%%%%%%%%% COVARIANCE MATRIX %%%%%%%%%%%%%%
c_mat = zeros(m, m);
for i = 1:n
    diff(:,i) = A.FACES(:,i) - face;
end
c_mat = diff'*diff;
c_mat = (1/9)*c_mat;

% %%%%%%%%%%%%% EIGEN VALUES & VECTORS %%%%%%%%%%
 [L_vect, D_values] = eig(c_mat);
 [D_values, ind] = sort(diag(D_values), 'descend');
 L_vect = L_vect(:, ind);
 e_vectors = diff*L_vect;
%  e_norml = norm(e_vectors);
%  e_vectors = (1/e_norml)*e_vectors;
 
for i = 1:n
    im_1{i} = e_vectors(:,i);     
    im_faces{i} = reshape(im_1{i}, [i_size, i_size]);
end

e_faces = struct('faces', []);
p = 1;
for i = 1: i_size: grid_size        %converting the image cells into structure
   for j = 1: i_size: grid_size
      e_faces.faces(i: i + (i_size-1) , j: j + (i_size-1)) = im_faces{p};
      p = p + 1;
   end
end
eigen_images = e_faces.faces;
figure(3); imshow(eigen_images, []);title('EIGEN IMAGES')

%%%%%%%%%%%%%%%%%RECONSTRUCTION %%%%%%%%%%%%%%%%%%%%%%%
num_eigenfaces = 6; %change the number here
new_e_vectors = zeros(m, n);
new_e_vectors (:, 1:num_eigenfaces)= e_vectors(:, 1:num_eigenfaces);
z = diff' * new_e_vectors; %projecting to a new feature space
%%%reconstruction
x_rec = z * new_e_vectors';
x_rec = x_rec';

for i = 1:n
    o{i} = x_rec(:,i) + face;
    output{i} = reshape(o{i}, [i_size, i_size]);
end

o_faces = struct('faces', []);
p = 1;
for i = 1: i_size: grid_size        %converting the image cells into structure
   for j = 1: i_size: grid_size
      o_faces.faces(i: i + (i_size-1) , j: j + (i_size-1)) = output{p};
      p = p + 1;
   end
end
output_images = o_faces.faces;
figure(5); imshow(output_images, []);title(['RECONSTRUCTED IMAGES WHEN #EIGENFACES =' num2str(num_eigenfaces)]);
