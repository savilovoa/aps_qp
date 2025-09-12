
count_days = 42
num_machines = 3
cleans = [(1, 6)]

weeks = range(count_days // 21)

work_days = []
work_days_week = [[] for w in weeks]
# Значение для отображения чистки в итоговом расписании
for m in range(num_machines):
    for d in range(count_days-1):
        if (m, d) not in cleans:
            work_days.append((m, d))
            # Домен переменной: от 0 до num_products - 1
            work_days_week[d // 21].append((m, d))

