В данном репозитории лежит выолненый кейс от Сириуса по теме computer vision. Вкратце нужно удалить фон, заменить его и затем описать объекты.
Для удаления фона использовалась готовая модель BriaAI. Ссылка на huggingface:
https://huggingface.co/briaai/RMBG-1.4
Для нахождения истинной маски производилось выделение контуров с помощью библиотеки opencv.
Так же для сегментации испольщовались:


Для расчета точности удаления фона будут сравниваться две готовые модели.
Используемые метрики: Intersection over union(IoU) и Dice coefficient(F1-score).


Для замены фона использовался градиентный фон, написанный собственноручно.
Выбор цветов замены фона: белый, серый, ярко-серый.
Для замены фона маски от готовой модели сохранялись в отдельный файл.


