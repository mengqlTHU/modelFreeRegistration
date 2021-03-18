filepath = "20200205_3axis";
matlab_log = fopen(strcat(filepath,"/","log_matlab.txt"),'r');
matlab_jp = cell2mat(textscan(matlab_log, "%f64,%f64,%f64,%f64,%f64,%f64,%f64\n"));

figure()
plot(0:delta_second:delta_second*(size(jRecords,1)-1), jRecords(:,5),'y-', 'LineWidth', 5);
hold on
plot(matlab_jp(:,1), matlab_jp(:,6), '--', 'LineWidth', 2);
xlabel('Time(s)');
ylabel('Joint Position (rad)');
legend('Expected Trajectory', 'Real Trajectory');
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',32)
