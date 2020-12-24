male=0;
female=0;
total=2000;
for i=4000:length(wiki.dob)
    if male>=total && female>=total
        break
    end
    fprintf('Copying face %d/%d\n',i,length(wiki.dob));
    if isnan(wiki.gender(i))
        ;
    elseif wiki.gender(i)==1 && male<=total
        copyfile(wiki.full_path{i},['male/' int2str(i) '.jpg']);
        male=male+1;
    elseif wiki.gender(i)==0 && female<=total
        copyfile(wiki.full_path{i},['female/' int2str(i) '.jpg']);
        female=female+1;
    end
end