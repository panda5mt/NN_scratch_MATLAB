
%backprop through maxpool layer

function [zz] = maxpool_backprop(data, m_pool, max_siz, stride, padding)

it2 = 0;

if sum(padding) >0
    data = padarray(data, padding,0);
end

zz = zeros(size(data));
for b = 1:size(data,4)
    it1 = 0;
for i = 1:stride(1):size(data,1)-1
    it1 = it1 + 1;
    for j = 1:stride(2):size(data,2)-1
        it2 = it2 + 1;
        for f = 1:size(data,3)
          
            r = data(i:(i+max_siz(1)-1), j:(j+max_siz(2)-1), f, b);
            z = zeros(size(r));
            z(r==max(r(:))) = m_pool(it1,it2,f,b); %making position of max equal to max and surrounding to 0
           
            zz(i:(i+max_siz(1)-1), j:(j+max_siz(1)-1), f, b) = zz(i:(i+max_siz(1)-1), j:(j+max_siz(1)-1), f, b) + z;
            
   
   
        end
    end
    it2 = 0;
end
end
end