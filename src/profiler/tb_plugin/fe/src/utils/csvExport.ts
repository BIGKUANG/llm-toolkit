/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export interface CsvColumn {
  title: string
  dataIndex?: string
  key?: string
}

interface ValueAndFormat {
  v: string | number | boolean
  f: string
}

function isValueAndFormat(value: any): value is ValueAndFormat {
  return value !== null && typeof value === 'object' && 'v' in value && 'f' in value
}

function extractValue(field: any): string | number | boolean | null | undefined {
  if (isValueAndFormat(field)) {
    return field.f || field.v
  }
  return field
}

function escapeCSVField(field: any): string {
  const value = extractValue(field)

  if (value === null || value === undefined) {
    return ''
  }

  const str = String(value)

  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`
  }

  return str
}

export function exportToCSV<T extends Record<string, any>>(
  columns: CsvColumn[],
  rows: T[],
  filename: string
): void {
  const headerRow = columns.map((col) => escapeCSVField(col.title)).join(',')

  const dataRows = rows.map((row) => {
    return columns
      .map((col) => {
        const key = col.dataIndex || col.key || ''
        const value = row[key]
        return escapeCSVField(value)
      })
      .join(',')
  })

  const csvContent = [headerRow, ...dataRows].join('\n')

  const BOM = '\uFEFF'
  const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' })

  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)

  URL.revokeObjectURL(url)
}

export function exportGraphToCSV(
  columns: Array<{ name: string; type?: string }>,
  rows: any[][],
  filename: string
): void {
  const headerRow = columns.map((col) => escapeCSVField(col.name)).join(',')

  const dataRows = rows.map((row) => {
    return row.map((cell) => escapeCSVField(cell)).join(',')
  })

  const csvContent = [headerRow, ...dataRows].join('\n')

  const BOM = '\uFEFF'
  const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' })

  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)

  URL.revokeObjectURL(url)
}

export function generateFilename(prefix: string): string {
  const now = new Date()
  const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19)
  return `${prefix}_${timestamp}.csv`
}
