# Сравнение локальной копии Morgan с GitHub

Проверено 6 ноября 2026 года. Snapshot фиксирует read-only сверку до публикации документов.
Работающий Morgan не менялся. Абсолютные локальные пути убраны; оригиналы сохранены у владельца.

## Репозиторий и основа следующей работы

- Репозиторий: [Morgan][source-1]
- Точный HTTPS remote для нового checkout: `https://github.com/lazarevtill/Morgan.git`
- Default branch: `main`, GitHub сообщает `protected: true`.
- Проверенный HEAD: `d29f53daa3c6eb6baef235c0d9888f959c286d13`, commit от 18.10.2026 22:13:06 UTC.
- Заголовок commit: `Retain explicit revisions and add opt-in consolidation byte limits (#69)`.
- Предлагаемая новая ветка для результатов исследования:
  `docs/personal-agent-24x7-research-2026-11-06`, от проверенного `main`. Это выбранное имя
  documentation branch для публикации исследования.

Исследованная папка `<local-research-copy>/Morgan` не содержит `.git`. У неё нельзя установить
branch/dirty status через git и нельзя утверждать, что у неё настроен origin. Адрес выше установлен
по локальным PR-артефактам и подтверждён GitHub API. Полный recursive tree текущего main получен без
truncation; сравнение выполнено по Git blob SHA-1 всех 319 файлов, а не по похожести названий.

## Различия содержимого

| Категория | Число |
| --- | ---: |
| Побайтово совпадают с main | 306 |
| Есть локально, но отличаются | 3 |
| Есть в main, отсутствуют локально | 10 |
| Только локально, без caches/pyc | 8 |

Три отличающихся файла:

- `morgan_brain/memory/knowledge/consolidation.py`: в main добавлен optional `request_byte_limit`,
  проверка параметра и передача лимита structured generation.
- `morgan_brain/providers/openai_compat.py`: в main добавлены immutable `ChatRequestOptions`, общий
  builder payload и opt-in управление output/template/temperature.
- `morgan_brain/providers/structured.py`: в main измеряется canonical UTF-8 JSON до каждого
  initial/re-ask запроса и отклоняется превышение optional ceiling.

Десять отсутствующих локально файлов:

- `docs/decisions/adapter-request-controls.md`
- `docs/decisions/consolidation-revision-byte-limit.md`
- `docs/decisions/reader-response-contract.md`
- `examples/consolidation_byte_limit_noop.py`
- `examples/reader_policy_zero_model.py`
- `morgan_brain/providers/request_budget.py`
- `tests/unit/providers/reader_contract_regression_proposal.json`
- `tests/unit/providers/test_reader_policy_example.py`
- `tests/unit/providers/test_request_byte_budget.py`
- `tests/unit/providers/test_request_options.py`

Все восемь local-only файлов — наши исследовательские материалы под
`docs/research/2026-11-06-personal-agent-memory/`. Иных local-only исходников относительно текущего
main не найдено. Сгенерированные caches/pyc исключены.

Дополнительная сверка с предыдущим main `b09bd13b9374b41b0a5d3587f722a50277edc535` показывает
отличия в consolidation.py и surprise.py и добавленный test_revision_surprise.py. Следовательно,
исследованная папка похожа на предыдущий main с частью работы по сохранению explicit revisions, а не
на чистый checkout одного доказанного commit. Без `.git` точную ancestry восстановить нельзя. В
текущем main surprise.py и test_revision_surprise.py уже совпадают с локальными.

## Слитые изменения и состояние CI

[PR #70][source-2] закрыт и merged в `fix/retain-revision-consolidation`; [PR #69][source-3] затем
merged в main с SHA `d29f53d…`. Не нужно открывать заново или дублировать эти изменения.

Все 19 check runs, возвращённых для текущего HEAD, завершены успешно. В том числе [Run Tests
py3.12][source-4] от 18.10.2026, lint/mypy, bandit, secret scanning и поздние CodeQL до 03.11.2026.
Это удалённые результаты CI этого SHA, не новый запуск тестов в данной сверке и не проверка
deployment. Перечень проверок и ссылок сохранён в `remote-state.json`.

Документ reader-response-contract уже находится в main, хотя начало ещё говорит `local candidate` и
`unpublished`. Это устаревший статус текста; фактический merge подтверждён API. Контракт различает
verified/unverified_report/unknown и не даёт execution authority. Новые byte controls по умолчанию
отключены; они не являются общим бюджетом токенов/расходов агента 24/7. Предшествующий recall может
выполнить embedding и запись fingerprint до отказа byte guard.

## Ветки и риск повторного использования

GitHub вернул десять веток:

| Ветка | HEAD | Значение |
| --- | --- | --- |
| `main` | `d29f53da` | Рекомендуемая основа новой работы |
| `feat/phase1a-capture-and-digest` | `7172a524` | Diverged: 32 ahead, 69 behind main; 54 файла в compare от merge base. Нельзя принимать за актуальную реализацию capture в main |
| `feat/working-context-v1` | `fa9f51d0` | Diverged: 39 ahead, 5 behind main; открытый PR #59; 43 файла в compare от merge base |
| `docs/long-lived-mac-memory` | `204190ab` | Открытый PR #63, отрицательный quality gate |
| `docs/budget-v2-negative-evaluation` | `dd06f59b` | Открытый PR #65 |
| `docs/public-continuation-negative` | `55606a1c` | Открытый PR #66 |
| `docs/reader-response-contracts` | `c7df8033` | Открытый PR #67; не путать с уже merged reader contract из #70 |
| `test/literal-evidence-resumption` | `6b03696e` | Открытый PR #68 |
| `feat/inspect-continuation-context` | `5d876906` | Ветка существует; не использовалась как основа |
| `test/bounded-memory-scale` | `fa77768b` | Ветка существует; не использовалась как основа |

Compare API для diverged веток показывает изменения от merge base, а не прямую diff между двумя
tips. Counts ahead/behind отражают commits; из них нельзя вывести, что готовая функциональность
отсутствует или корректна без просмотра реализации.

На capture-and-digest есть изменения config, MemoryGate, module, migrations, session/digest/call
stores, secret scanning и тесты. Это крупный исторический набор, а не маленький reader разговоров.
Его можно использовать как источник решений и тестов в отдельной проработке, но перенос целой ветки
создаёт риск вернуть старые схемы и отменить новые гарантии main. Working-context также не следует
включать ради публикации исследования.

## Что меняется в выводах исследования

Основной вывод сохраняется: актуальные README/ROADMAP, MemoryGate, FTS, exact evidence и importer
совпадают с исследованной копией. Main остаётся ядром памяти; готовый runtime персонального агента
24/7 здесь не появился. Но опубликованные request controls и reader-policy пример уже следует
считать реализованными opt-in средствами, а не будущей работой.

Ранее выполненные 78 тестов относились к изолированной исследованной копии. Они не проверяли
отсутствовавшие request-budget/reader-policy функции. Нельзя переподписать их как новый полный test
result актуального main. Если будут code changes, проверять нужно свежий checkout и затронутые новые
тесты, затем необходимые repository checks.

## Безопасный путь внесения результатов

1. Для дальнейших правок создать отдельный настоящий checkout репозитория с main, зафиксировать
   актуальный SHA повторным чтением и убедиться в чистом status. Не выполнять git init поверх
   исследованной папки и не копировать её целиком на main.
2. Создать новую documentation branch от этого main. Перенести исследовательский Markdown,
   `sources.json`, `sources-runtime.json` и этот addendum,
   сохранив исходный отчёт как dated snapshot.
   Добавить короткую ссылку из
   Documentation/ROADMAP и отдельно сформулировать новую цель владельца: персональный агент 24/7
   поверх независимого memory core. Не объявлять runtime реализованным.
3. До будущей публичной публикации проверить абсолютные локальные пути и журналы; remote публичный.
   Не переносить автоматически raw test logs, окружение, локальные manifests и копии исходников. Они
   полезны локально, но не обязательны в публичном PR.
4. Не смешивать documentation change с merge старых веток, миграцией БД, установкой runtime или
   запуском служб. Runtime integration оформлять отдельным ограниченным этапом с
   recovery/cancellation/budget тестами на synthetic data.
5. После сверки владелец отдельно разрешил публикацию документации через указанную ветку, PR и merge
   при успешных проверках. Это не разрешение запускать постоянного агента или изменять deployment.

Следующие supporting artifacts сохранены только локально и не входят в публичный PR:
`main-tree.json`, `previous-main-tree.json`, `comparison.json`, `remote-state.json`,
`local-to-main.diff` и три файла текущего main в `current-main/`. Исходный исследовательский
REPORT.md, включая уже загруженную Library-версию, сохранён без изменений.

[source-1]: https://github.com/lazarevtill/Morgan
[source-2]: https://github.com/lazarevtill/Morgan/pull/70
[source-3]: https://github.com/lazarevtill/Morgan/pull/69
[source-4]: https://github.com/lazarevtill/Morgan/actions/runs/39171260549/job/117804620988
