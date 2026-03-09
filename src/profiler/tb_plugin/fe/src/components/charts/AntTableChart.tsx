/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import Button from '@material-ui/core/Button'
import GetAppIcon from '@material-ui/icons/GetApp'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Radio from '@material-ui/core/Radio'
import RadioGroup from '@material-ui/core/RadioGroup'
import { Table } from 'antd'
import * as React from 'react'
import { Graph } from '../../api'
import { exportGraphToCSV, generateFilename } from '../../utils/csvExport'

type TimeUnit = 'us' | 'ms'

function isTimeColumn(columnName: string): boolean {
  const lowerName = columnName.toLowerCase()
  return lowerName.includes('time') || lowerName.includes('duration')
}

function convertTimeValue(value: number | undefined | null, unit: TimeUnit): string {
  if (value === undefined || value === null || typeof value !== 'number') {
    return String(value ?? '')
  }
  if (unit === 'ms') {
    return (value / 1000).toFixed(4)
  }
  return value.toFixed(4)
}

interface IProps {
  graph: Graph
  sortColumn?: string
  initialPageSize?: number
  onRowSelected?: (record?: object, rowIndex?: number) => void
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap'
  },
  row: {
    wordBreak: 'break-word'
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

const getTableColumns = function (
  columns: any,
  sort: string | undefined,
  tooltipClass: string,
  timeUnit: TimeUnit
) {
  let i = 0
  return columns.map(function (col: any) {
    const key = 'col' + i++
    const stringCompare = (a: any, b: any) => a[key].localeCompare(b[key])
    const numberCompare = (a: any, b: any) => (a[key] || 0) - (b[key] || 0)
    const isTime = isTimeColumn(col.name)
    const displayTitle = isTime ? col.name.replace(/\(us\)/gi, `(${timeUnit})`) : col.name
    
    const columnDef: any = {
      dataIndex: key,
      key: key,
      title: displayTitle,
      sorter: col.type == 'string' ? stringCompare : numberCompare,
      defaultSortOrder: sort == col.name ? ('descend' as const) : undefined,
      showSorterTooltip: col.tooltip
        ? { title: col.tooltip, overlayClassName: tooltipClass }
        : true
    }
    
    if (isTime && col.type !== 'string') {
      columnDef.render = (value: number) => convertTimeValue(value, timeUnit)
    }
    
    return columnDef
  })
}

const getTableRows = function (rows: any) {
  return rows.map(function (row: any) {
    let i = 0
    const res: any = {}
    row.forEach(function (entry: any) {
      res['col' + i++] = entry
    })
    return res
  })
}

const ALL_PAGE_SIZE = -1

export const AntTableChart: React.FC<IProps> = (props) => {
  const { graph, sortColumn, initialPageSize, onRowSelected } = props
  const classes = useStyles(props)

  const [timeUnit, setTimeUnit] = React.useState<TimeUnit>('us')

  const rows = React.useMemo(() => getTableRows(graph.rows), [graph.rows])

  const columns = React.useMemo(
    () => getTableColumns(graph.columns, sortColumn, classes.tooltip, timeUnit),
    [graph.columns, sortColumn, classes.tooltip, timeUnit]
  )

  // key is used to reset the Table state (page and sort) if the columns change
  const key = React.useMemo(() => Math.random() + '', [graph.columns, timeUnit])

  const [pageSize, setPageSize] = React.useState(initialPageSize ?? 30)
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size)
  }

  const actualPageSize = pageSize === ALL_PAGE_SIZE ? rows.length : pageSize

  const handleTimeUnitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTimeUnit(event.target.value as TimeUnit)
  }

  const handleDownloadCSV = React.useCallback(() => {
    const filename = generateFilename('kernel')
    exportGraphToCSV(graph.columns, graph.rows, filename)
  }, [graph.columns, graph.rows])

  const onRow = (record: object, rowIndex?: number) => {
    return {
      onMouseEnter: (event: any) => {
        if (onRowSelected) {
          onRowSelected(record, rowIndex)
        }
      },
      onMouseLeave: (event: any) => {
        if (onRowSelected) {
          onRowSelected(undefined, undefined)
        }
      }
    }
  }

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
        rowClassName={classes.row}
        key={key}
        onRow={onRow}
      />
    </div>
  )
}
