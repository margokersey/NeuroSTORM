folder = "/Users/mkersey/Downloads/Task timing"; % Download from DropBox/HuangLab/Database/Michigan_Dataset1/Task Timing

suffix = ["action","navi","relax","squeeze","tennis"];

delay = 0; % delay should be set to 0, not 4 as it was previously;

len = ([10 15 30 30 15 10].*60)+16;
TR_len = len ./ 0.8;

number_rest1 = ones(len(1),1);
letter_rest1 = repmat("rest",len(1),1);

%%
number_task1 = ones(len(2),1);
letter_task1 = repmat("rest",len(2),1);

list = [];
for fn = 1:numel(suffix)
    s = readmatrix(folder+"/task1_"+suffix(fn)+".1D",'FileType','text');
    for i = 1:numel(s)
        if ~isnan(s(i))
            tmp = {suffix(fn),s(i)};
            list = [list;tmp];
        end
    end
end

for i = 1:length(list)
    task_name = list{i,1};
    idx = list{i,2};
    switch task_name
        case "squeeze"
            number_task1(idx+1:idx+10) = 2;
            letter_task1(idx+1:idx+10) = "squeeze imagery";
        case "action"
            number_task1(idx+1:idx+6) = 3;
            letter_task1(idx+1:idx+6) = "squeeze action";
        case "navi"
            number_task1(idx+1:idx+16) = 4;
            letter_task1(idx+1:idx+16) = "navigation imagery";
        case "tennis"
            number_task1(idx+1:idx+16) = 5;
            letter_task1(idx+1:idx+16) = "tennis imagery";
    end
end

%%
number_task2 = ones(len(3),1);
letter_task2 = repmat("rest",len(3),1);

list = [];
for fn = 1:numel(suffix)
    s = readmatrix(folder+"/task2_"+suffix(fn)+".1D",'FileType','text');
    for i = 1:numel(s)
        if ~isnan(s(i))
            tmp = {suffix(fn),s(i)};
            list = [list;tmp];
        end
    end
end

for i = 1:length(list)
    task_name = list{i,1};
    idx = list{i,2};
    switch task_name
        case "squeeze"
            number_task2(idx+1:idx+10) = 2;
            letter_task2(idx+1:idx+10) = "squeeze imagery";
        case "action"
            number_task2(idx+1:idx+6) = 3;
            letter_task2(idx+1:idx+6) = "squeeze action";
        case "navi"
            number_task2(idx+1:idx+16) = 4;
            letter_task2(idx+1:idx+16) = "navigation imagery";
        case "tennis"
            number_task2(idx+1:idx+16) = 5;
            letter_task2(idx+1:idx+16) = "tennis imagery";
    end
end

%%
number_task3 = ones(len(4),1);
letter_task3 = repmat("rest",len(4),1);

list = [];
for fn = 1:numel(suffix)
    s = readmatrix(folder+"/task3_"+suffix(fn)+".1D",'FileType','text');
    for i = 1:numel(s)
        if ~isnan(s(i))
            tmp = {suffix(fn),s(i)};
            list = [list;tmp];
        end
    end
end

for i = 1:length(list)
    task_name = list{i,1};
    idx = list{i,2};
    switch task_name
        case "squeeze"
            number_task3(idx+1:idx+10) = 2;
            letter_task3(idx+1:idx+10) = "squeeze imagery";
        case "action"
            number_task3(idx+1:idx+6) = 3;
            letter_task3(idx+1:idx+6) = "squeeze action";
        case "navi"
            number_task3(idx+1:idx+16) = 4;
            letter_task3(idx+1:idx+16) = "navigation imagery";
        case "tennis"
            number_task3(idx+1:idx+16) = 5;
            letter_task3(idx+1:idx+16) = "tennis imagery";
    end
end

%%
number_task4 = ones(len(5),1);
letter_task4 = repmat("rest",len(5),1);

list = [];
for fn = 1:numel(suffix)
    s = readmatrix(folder+"/task4_"+suffix(fn)+".1D",'FileType','text');
    for i = 1:numel(s)
        if ~isnan(s(i))
            tmp = {suffix(fn),s(i)};
            list = [list;tmp];
        end
    end
end

for i = 1:length(list)
    task_name = list{i,1};
    idx = list{i,2};
    switch task_name
        case "squeeze"
            number_task4(idx+1:idx+10) = 2;
            letter_task4(idx+1:idx+10) = "squeeze imagery";
        case "action"
            number_task4(idx+1:idx+6) = 3;
            letter_task4(idx+1:idx+6) = "squeeze action";
        case "navi"
            number_task4(idx+1:idx+16) = 4;
            letter_task4(idx+1:idx+16) = "navigation imagery";
        case "tennis"
            number_task4(idx+1:idx+16) = 5;
            letter_task4(idx+1:idx+16) = "tennis imagery";
    end
end

%%
number_rest2 = ones(len(6),1);
letter_rest2 = repmat("rest",len(6),1);


%%
number_merge = [ones(delay,1); number_rest1; number_task1; number_task2; number_task3; number_task4; number_rest2(delay+1:end)];
letter_merge = [repmat("rest",delay,1); letter_rest1; letter_task1; letter_task2; letter_task3; letter_task4; letter_rest2(delay+1:end)];

idx = 1:8370;
idx = round(idx .* 0.8);

for i = 1:numel(idx)
    number(i) = number_merge(idx(i));
    letter(i) = letter_merge(idx(i));
end

save("timing.mat","letter","number");

%%

rest = zeros(numel(number),1);
squeeze_imagery = rest;
squeeze_action = rest;
navigation = rest;
tennis = rest;

for i = 1:numel(number)
    switch number(i)
        case 1
            rest(i) = 1;
        case 2
            squeeze_imagery(i) = 1;
        case 3
            squeeze_action(i) = 1;
        case 4
            navigation(i) = 1;
        case 5
            tennis(i) = 1;
    end
end 

T = table(rest,squeeze_imagery,squeeze_action,navigation,tennis);

%%

conscious = zeros(numel(number),1);
unconscious = conscious;

for i = 1:numel(number)
    switch number(i)
        case 1
            rest(i) = 1;
        case 2
            squeeze_imagery(i) = 1;
        case 3
            squeeze_action(i) = 1;
        case 4
            navigation(i) = 1;
        case 5
            tennis(i) = 1;
    end
end 

T = table(rest,squeeze_imagery,squeeze_action,navigation,tennis);


writetable(T,'timing_onehot.tsv','FileType','text','Delimiter','\t');
writetable(T(1:1915,:),'timing_onehot_rest1_task1.tsv','FileType','text','Delimiter','\t');
writetable(T(771:1915,:),'timing_onehot_task1.tsv','FileType','text','Delimiter','\t');
%writetable(T(1916:1915,:),'timing_onehot_task1.tsv','FileType','text','Delimiter','\t');


% Added by margo

disp(size(T));

% Rest 1: 770 TR
writetable(T(1:770,:),'timing_onehot_rest1.tsv','FileType','text','Delimiter','\t');

% Task 1: 1145 TR
writetable(T(771:1915,:),'timing_onehot_task1.tsv','FileType','text','Delimiter','\t');

% Task 2: 2270 TR
writetable(T(1916:4185,:),'timing_onehot_task2.tsv','FileType','text','Delimiter','\t');

% Task 3: 2270 TR
writetable(T(4186:6455,:),'timing_onehot_task3.tsv','FileType','text','Delimiter','\t');

% Task 4: 1145 TR
writetable(T(6456:7600,:),'timing_onehot_task4.tsv','FileType','text','Delimiter','\t');

% Rest 2: 770
writetable(T(7601:8370,:),'timing_onehot_rest2.tsv','FileType','text','Delimiter','\t');

