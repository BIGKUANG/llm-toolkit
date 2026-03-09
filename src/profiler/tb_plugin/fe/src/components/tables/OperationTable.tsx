/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import Button from '@material-ui/core/Button'
import GetAppIcon from '@material-ui/icons/GetApp'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Radio from '@material-ui/core/Radio'
import RadioGroup from '@material-ui/core/RadioGroup'
import {
  OperationTableData,
  OperationTableDataInner
} from '../../api'
import { OperationGroupBy } from '../../constants/groupBy'
import { attachId, getCommonOperationColumns, TimeUnit } from './common'
import { Table, TableProps } from 'antd'
import { makeExpandIcon } from './ExpandIcon'
import { CallStackTable } from './CallStackTable'
import { exportToCSV, generateFilename, CsvColumn } from '../../utils/csvExport'

export interface IProps {
  data: OperationTableData
  run: string
  worker: string
  span: string
  groupBy: OperationGroupBy
  sortColumn: string
  tooltips?: any
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap'
  },
  toolbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing(1)
  },
  toolbarLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1)
  },
  toolbarRight: {
    display: 'flex',
    alignItems: 'center'
  },
  downloadButton: {
    textTransform: 'none'
  },
  unitLabel: {
    marginRight: theme.spacing(1),
    fontSize: '14px',
    color: theme.palette.text.secondary
  },
  radioGroup: {
    flexDirection: 'row'
  }
}))

const rowExpandable = (record: OperationTableDataInner) => record.has_call_stack
const expandIcon = makeExpandIcon<OperationTableDataInner>(
  'View CallStack',
  (record) => !record.has_call_stack
)
const ALL_PAGE_SIZE = -1

export const OperationTable = (props: IProps) => {
  const { data, run, worker, span, groupBy, sortColumn, tooltips } = props
  const classes = useStyles(props)

  const [timeUnit, setTimeUnit] = React.useState<TimeUnit>('us')

  const rows = React.useMemo(() => attachId(data), [data])

  const columns = React.useMemo(
    () => getCommonOperationColumns(rows, sortColumn, tooltips, classes, timeUnit),
    [rows, sortColumn, tooltips, classes, timeUnit]
  )

  const [pageSize, setPageSize] = React.useState(30)
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size)
  }

  const actualPageSize = pageSize === ALL_PAGE_SIZE ? rows.length : pageSize

  const handleTimeUnitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTimeUnit(event.target.value as TimeUnit)
  }

  const handleDownloadCSV = React.useCallback(() => {
    const csvColumns: CsvColumn[] = columns.map((col: any) => ({
      title: String(col.title || ''),
      dataIndex: col.dataIndex as string,
      key: col.key as string
    }))
    const filename = generateFilename('operator')
    exportToCSV(csvColumns, rows, filename)
  }, [columns, rows])

  const expandIconColumnIndex = columns.length
  const expandedRowRender = React.useCallback(
    (record: OperationTableDataInner) => (
      <CallStackTable
        data={record}
        run={run}
        worker={worker}
        span={span}
        groupBy={groupBy}
      />
    ),
    [run, worker, span, groupBy]
  )

  const expandable: TableProps<OperationTableDataInner>['expandable'] = React.useMemo(
    () => ({
      expandIconColumnIndex,
      expandIcon,
      expandedRowRender,
      rowExpandable
    }),
    [expandIconColumnIndex, expandedRowRender]
  )

  return (
    <div>
      <div className={classes.toolbar}>
        <div className={classes.toolbarLeft}>
          <span className={classes.unitLabel}>时间单位:</span>
          <RadioGroup
            value={timeUnit}
            onChange={handleTimeUnitChange}
            className={classes.radioGroup}
          >
            <FormControlLabel value="us" control={<Radio size="small" />} label="us" />
            <FormControlLabel value="ms" control={<Radio size="small" />} label="ms" />
          </RadioGroup>
        </div>
        <div className={classes.toolbarRight}>
          <Button
            variant="outlined"
            size="small"
            className={classes.downloadButton}
            startIcon={<GetAppIcon />}
            onClick={handleDownloadCSV}
          >
            下载 CSV
          </Button>
        </div>
      </div>
      <Table
        size="small"
        bordered
        columns={columns}
        dataSource={rows}
        pagination={{
          pageSize: actualPageSize,
          pageSizeOptions: ['10', '20', '30', '50', '100', `${rows.length}`],
          showSizeChanger: true,
          showTotal: (total: number) => `共 ${total} 条`,
          onShowSizeChange,
          locale: {
            items_per_page: '条/页'
          }
        }}
        expandable={expandable}
      />
    </div>
  )
}
