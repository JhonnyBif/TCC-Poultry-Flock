import { distinctUntilChanged, lastValueFrom, Subject, takeUntil } from 'rxjs';
import { CommonModule } from '@angular/common';
import {
  Component,
  OnDestroy,
  OnInit,
  TemplateRef,
  ViewChild,
} from '@angular/core';
import {
  FormControl,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { MatError, MatFormFieldModule } from '@angular/material/form-field';
import { MatRadioModule } from '@angular/material/radio';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatInputModule } from '@angular/material/input';
import { MatCardModule } from '@angular/material/card';
import { NgxFileDropModule } from 'ngx-file-drop';
import { MatListModule } from '@angular/material/list';
import { MatTableModule } from '@angular/material/table';
import {
  MatDialog,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
import { UploadService } from '../../shared/services/upload.service';
import { ModalResultadoComponent } from '../../shared/modais/modal-resultado/modal-resultado.component';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { PontoSelectorModalComponent } from '../../shared/modais/ponto-selector-modal/ponto-selector-modal.component';
import { FilePreview } from '../../shared/interfaces/file-preview.interface';
import { Router, RouterModule } from '@angular/router';
import { BreakpointObserver } from '@angular/cdk/layout';
import { MatSelectModule } from '@angular/material/select';
import { MedidasForm } from './forms/medidas.forms';
import { MatDatepickerModule } from '@angular/material/datepicker';
import {
  MAT_DATE_LOCALE,
  provideNativeDateAdapter,
} from '@angular/material/core';
import * as _moment from 'moment';
import { default as _rollupMoment } from 'moment';
import 'moment/locale/pt-br';
const moment = _rollupMoment || _moment;

@Component({
  selector: 'app-video-processor',
  standalone: true,
  imports: [
    FormsModule,
    CommonModule,
    MatRadioModule,
    MatIconModule,
    MatButtonModule,
    MatFormFieldModule,
    MatCheckboxModule,
    MatInputModule,
    NgxFileDropModule,
    MatCardModule,
    MatListModule,
    MatError,
    MatTableModule,
    MatProgressSpinnerModule,
    RouterModule,
    MatSelectModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatDatepickerModule,
  ],
  providers: [
    provideNativeDateAdapter(),
    { provide: MAT_DATE_LOCALE, useValue: 'pt-BR' },
  ],
  templateUrl: './video-processor.component.html',
  styleUrl: './video-processor.component.scss',
})
export class VideoProcessorComponent implements OnInit, OnDestroy {
  selectedOption: string = '';
  options = ['image', 'fragment', 'video', 'location'];
  optionsAglomeracao = ['Datas', 'Idade'];
  selectedOptionAglomeracao: string = '';
  parteCima: boolean = false;
  parteBaixo: boolean = false;
  selecionarMinutos: boolean = false;
  inicio: string = '';
  fim: string = '';
  files: FilePreview[] = [];
  error: boolean = false;
  carregando: boolean = false;
  tipoArquivo: 'image' | 'video' | null = null;
  pontos: { x: number; y: number }[] = [];
  requestError: boolean = false;
  noFilesError: boolean = false;
  noDateError: boolean = false;
  isMobile: boolean = false;
  datasets = new FormControl<string[]>(['Detectar Quantidade Por Metro Quadrado']);
  @ViewChild('meuDialogTemplate') meuDialogTemplate!: TemplateRef<any>;
  medidasForm = new MedidasForm();
  dialogRef!: MatDialogRef<any>;
  videoError: string = '';

  private destroy$ = new Subject<void>();

  constructor(
    private dialog: MatDialog,
    private uploadService: UploadService,
    private breakpointObserver: BreakpointObserver,
  ) {
    this.isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  }

  async ngOnInit() {
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }



  convertFileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
    });
  }



  cameraSelected(event: any): void {
    const file: File = event.target.files[0];
    if (!file) return;

    if (file.size <= 150 * 1024 * 1024) {
      const isImage = file.type.startsWith('image');
      if (!isImage) return;

      const url = URL.createObjectURL(file);
      this.files.push({ file, url, type: 'image' });
      this.tipoArquivo = 'image';
      this.error = false;
    } else {
      this.error = true;
    }
  }

  enviarMedidas() {
    this.dialog.closeAll();
  }

  async processar() {
    this.videoError = '';

    if (this.files.length === 0 || this.selectedOption == '') {
      this.noFilesError = true;
      setTimeout(() => {
        this.noFilesError = false;
        return;
      }, 1500);
      return;
    }
    let regionPoints: { x: number; y: number }[] | null = null;
    if (this.datasets.value?.includes('Detectar Movimentação')) {
      this.selectedOption = this.options[3];
    }
    if (
      this.datasets.value?.includes('Detectar Quantidade Por Metro Quadrado')
    ) {
      this.dialogRef = this.dialog.open(this.meuDialogTemplate);
      await lastValueFrom(this.dialogRef.afterClosed());
      if (this.medidasForm.invalid) {
        this.carregando = false;
        return;
      }
    }

    this.carregando = true;
    this.uploadService
      .enviarVideo(
        this.files[0],
        this.selectedOption,
        this.parteCima,
        this.parteBaixo,
        regionPoints,
        this.medidasForm.value,
        this.datasets.value,
      )
      .subscribe({
        next: (res) => {
          this.carregando = false;
          const {
            quantidade_por_metro_quadrado,
          } = res;
          this.abrirModal(quantidade_por_metro_quadrado,)
        },
        error: (err) => {
          this.carregando = false;
          this.requestError = true;
          setTimeout(() => {
            this.requestError = false;
          }, 2000);
          console.error('Erro ao enviar:', err);
        },
      });
  }

  abrirModal(
    quantidade_por_metro_quadrado: any,
  ): void {
    const isMobile = this.breakpointObserver.isMatched('(max-width: 600px)');
    this.dialog.open(ModalResultadoComponent, {
      width: isMobile ? '100vw' : '90vw',
      minWidth: isMobile ? '100vw' : '90vw',
      maxWidth: isMobile ? '100vw' : '90vw',
      height: '90vh',
      panelClass: isMobile ? 'full-screen-modal' : 'custom-modal',
      data: { quantidade_por_metro_quadrado},
    });
  }

  dropped(event: any): void {
    const droppedFiles = event;
    for (const droppedFile of droppedFiles) {
      droppedFile.fileEntry.file((file: File) => {
        if (file.size <= 150 * 1024 * 1024) {
          const isImage = file.type.startsWith('image');
          const isVideo = file.type.startsWith('video');
          this.tipoArquivo = isImage ? 'image' : isVideo ? 'video' : null;

          if (!isImage && !isVideo) return;

          const url = URL.createObjectURL(file);
          const type = isImage ? 'image' : 'video';

          this.files.push({ file, url, type });
          this.error = false;
        } else {
          this.error = true;
        }
      });
    }
  }

  removeFile(index: number): void {
    const item = this.files[index];
    if (item.url) {
      URL.revokeObjectURL(item.url);
    }
    this.files.splice(index, 1);
  }

  async convertToBase64(file: File): Promise<string | null> {
    return new Promise((resolve) => {
      const reader = new FileReader();

      reader.onload = (event: any) => {
        resolve(event.target.result);
      };

      if (file.type.match(/image.*/) || file.type.match(/pdf.*/)) {
        reader.readAsDataURL(file);
      } else {
        resolve(null);
      }
    });
  }

  bytesToMegabytes(bytes: number): string {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  }

  formataNomeArquivo(texto: string): string {
    return texto.length > 20
      ? texto.charAt(0).toUpperCase() + texto.slice(1, 20) + '...'
      : texto.charAt(0).toUpperCase() + texto.slice(1);
  }

  onNumberInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const controlName = input.name;
    input.value = input.value.replace(',', '.');
    input.value = input.value.replace(/[^0-9.]/g, '');
    const parts = input.value.split('.');
    if (parts.length > 2) {
      input.value = parts[0] + '.' + parts.slice(1).join('');
    }
    this.medidasForm
      .get(controlName)
      ?.setValue(input.value, { emitEvent: false });
  }

  preventInvalidKeys(event: KeyboardEvent): void {
    const invalidKeys = ['-', 'e', '+'];
    if (invalidKeys.includes(event.key)) {
      event.preventDefault();
    }
  }

  onOptionSelected(event: any): void {
    console.log(event);
    const dataEntrada = this.medidasForm.get('dataEntrada')!;
    const dataVideo = this.medidasForm.get('dataVideo')!;
    const idade = this.medidasForm.get('idade')!;
    if (this.selectedOptionAglomeracao === 'Idade') {
      dataVideo.setValidators(null);
      dataVideo.setValue(null);
      dataVideo.updateValueAndValidity();
      dataEntrada.setValidators(null);
      dataEntrada.setValue(null);
      dataEntrada.updateValueAndValidity();
      idade.setValidators(Validators.required);
      idade.updateValueAndValidity();
    } else {
      idade.setValidators(null);
      idade.setValue(null);
      idade.updateValueAndValidity();
      dataVideo.setValidators(Validators.required);
      dataVideo.updateValueAndValidity();
      dataEntrada.setValidators(Validators.required);
      dataEntrada.updateValueAndValidity();
    }
  }

  formatDateTime(date: Date, time: string): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');

    const [hours, minutes] = time.split(':').map(Number);

    const combinedDate = new Date(
      year,
      date.getMonth(),
      date.getDate(),
      hours,
      minutes,
      0,
      0
    );

    const formatted = `${combinedDate.getFullYear()}-${String(
      combinedDate.getMonth() + 1
    ).padStart(2, '0')}-${String(combinedDate.getDate()).padStart(
      2,
      '0'
    )} ${String(combinedDate.getHours()).padStart(2, '0')}:${String(
      combinedDate.getMinutes()
    ).padStart(2, '0')}:00.000`;

    return formatted;
  }
}
