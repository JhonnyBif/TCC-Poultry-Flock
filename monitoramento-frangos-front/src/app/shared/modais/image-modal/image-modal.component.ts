import { CommonModule } from '@angular/common';
import {
  Component,
  computed,
  inject,
  Inject,
  model,
  signal,
} from '@angular/core';
import { FormControl, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatCheckboxModule } from '@angular/material/checkbox';
import {
  MAT_DIALOG_DATA,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatRadioModule } from '@angular/material/radio';
import { MatTableModule } from '@angular/material/table';
import { NgxFileDropModule } from 'ngx-file-drop';
import { MatChipInputEvent, MatChipsModule } from '@angular/material/chips';
import {
  MatAutocompleteModule,
  MatAutocompleteSelectedEvent,
} from '@angular/material/autocomplete';
import { LiveAnnouncer } from '@angular/cdk/a11y';
import { COMMA, ENTER } from '@angular/cdk/keycodes';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MAT_DATE_LOCALE } from '@angular/material/core';
import * as _moment from 'moment';
import { default as _rollupMoment } from 'moment';
import 'moment/locale/pt-br';
import { provideMomentDateAdapter } from '@angular/material-moment-adapter';
import { UploadService } from '../../services/upload.service';
const moment = _rollupMoment || _moment;

@Component({
  selector: 'app-image-modal',
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
    MatTableModule,
    MatProgressSpinnerModule,
    MatDialogModule,
    MatChipsModule,
    MatAutocompleteModule,
    MatDatepickerModule,
    ReactiveFormsModule,
  ],
  providers: [
    provideMomentDateAdapter(),
    { provide: MAT_DATE_LOCALE, useValue: 'pt-BR' },
  ],
  templateUrl: './image-modal.component.html',
  styleUrls: ['./image-modal.component.scss'],
})
export class ImageModalComponent {
  pontos: { x: number; y: number }[] = [];
  observacao = '';
  mostrarFormulario: boolean = false;
  observacoesSalvas: {
    pontos: { x: number; y: number }[];
    texto: string;
    infos: string;
    data: string;
    idade: string;
    peso: string;
    image: string;
  }[] = [];

  informacoes = signal([] as string[]);
  readonly separatorKeysCodes: number[] = [ENTER, COMMA];
  readonly currentInformacao = model('');
  readonly allInformacoes: string[] = [
    'Macho',
    'Fêmea',
    'Calor',
    'Aglomeração',
    'Peso',
  ];
  readonly filteredInformacoes = computed(() => {
    const currentFruit = this.currentInformacao().toLowerCase();
    return currentFruit
      ? this.allInformacoes.filter((informacao) =>
          informacao.toLowerCase().includes(currentFruit)
        )
      : this.allInformacoes.slice();
  });
  readonly date = new FormControl(moment());
  readonly announcer = inject(LiveAnnouncer);
  idadeFrango = new FormControl(0);
  pesoFrango = new FormControl(0);
  carregando: boolean = false;
  requestError: boolean = false;
  requestValid: boolean = false;

  constructor(
    @Inject(MAT_DIALOG_DATA) public data: { imagem: string },
    private dialogRef: MatDialogRef<ImageModalComponent>,
    private readonly uploadService: UploadService
  ) {}

  registrarClique(event: MouseEvent) {
    const imgElement = event.target as HTMLElement;
    const rect = imgElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (this.pontos.length < 4) {
      this.pontos.push({ x: Math.round(x), y: Math.round(y) });
      if (this.pontos.length === 4) {
        this.mostrarFormulario = true;
      }
    }
  }

  get pontosComoString(): string {
    return this.pontos.map((p) => `${p.x},${p.y}`).join(' ');
  }

  add(event: MatChipInputEvent): void {
    const value = (event.value || '').trim();
    if (value) {
      this.informacoes.update((informacoes) => [...informacoes, value]);
    }
    this.currentInformacao.set('');
  }

  addManual(): void {
    const value = this.currentInformacao().trim();
    if (value) {
      this.informacoes.update((informacoes) => [...informacoes, value]);
      this.currentInformacao.set('');
    }
  }

  remove(informacao: string): void {
    this.informacoes.update((informacoes) => {
      const index = informacoes.indexOf(informacao);
      if (index < 0) {
        return informacoes;
      }

      informacoes.splice(index, 1);
      this.announcer.announce(`Removed ${informacao}`);
      return [...informacoes];
    });
  }

  selected(event: MatAutocompleteSelectedEvent): void {
    this.informacoes.update((informacoes) => [
      ...informacoes,
      event.option.viewValue,
    ]);
    this.currentInformacao.set('');
    event.option.deselect();
  }

  onlyNumberAllowed(event: KeyboardEvent): void {
    const teclasPermitidas = [
      'Backspace',
      'ArrowLeft',
      'ArrowRight',
      'Tab',
      'Delete',
    ];
    if (event.ctrlKey || event.metaKey) return;
    if (!/^[0-9]$/.test(event.key) && !teclasPermitidas.includes(event.key)) {
      event.preventDefault();
    }
  }

  cancelarObservacao() {
    this.currentInformacao.set('');
    this.date.setValue(moment());
    this.idadeFrango = new FormControl(0);
    this.pesoFrango = new FormControl(0);
    this.pontos = [];
    this.observacao = '';
    this.mostrarFormulario = false;
  }

  async salvarObservacao() {
    this.observacoesSalvas.push({
      pontos: [...this.pontos],
      infos: this.informacoes().toString(),
      data: this.date.value!.toString(),
      idade: this.isValorValido(this.idadeFrango)
        ? this.idadeFrango.toString()
        : '',
      peso: this.isValorValido(this.pesoFrango)
        ? this.pesoFrango.toString()
        : '',
      texto: this.observacao.toString(),
      image: this.data.imagem,
    });
    this.informacoes = signal([] as string[]);
    this.currentInformacao.set('');
    this.date.setValue(moment());
    this.idadeFrango = new FormControl(0);
    this.pesoFrango = new FormControl(0);
    this.pontos = [];
    this.observacao = '';
    this.mostrarFormulario = false;
  }

  isValorValido(valor: any): boolean {
    return typeof valor === 'number' || typeof valor === 'string';
  }

  async save() {
    this.carregando = true;
    this.uploadService
      .salvarImagemParaTreinamento(this.observacoesSalvas)
      .subscribe({
        next: (res: any) => {
          this.requestValid = true;
          this.carregando = false;
          setTimeout(() => {
            this.requestValid = false;
            this.dialogRef.close();
          }, 2000);
        },
        error: (err: any) => {
          this.carregando = false;
          this.requestError = true;
          setTimeout(() => {
            this.requestError = false;
          }, 2000);
          console.error('Erro ao enviar:', err);
        },
      });
  }
}
